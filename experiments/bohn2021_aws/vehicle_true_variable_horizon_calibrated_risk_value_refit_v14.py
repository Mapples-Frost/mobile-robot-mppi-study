#!/usr/bin/env python3
"""v14 calibrated risk/value refit for true-variable-H H10/H15.

Development-only IMPROVED diagnostic after v13b.  v11/v12 established that
opened branch states contain a real H10/H15 control-compute opportunity, while
v13/v13b showed that static/H15-prefix support selectors do not safely deploy it.
This script tests a more explicit hypothesis: a deployable selector that predicts
catastrophic-H10 risk and H10-vs-H15 continuation value with calibrated
uncertainty can abstain on ambiguous states while preserving useful measured
branch decision-time saving.

It consumes only already-opened development banks fresh_v0/fresh_v1/fresh_v2/
fresh_v8c/fresh_v11 through v13's loader and v12's outcome-aligned labels.  It
performs no MPC simulation, no gradient/RL training, no validation64 access, and
no sealed-test access.  Scenario group/window, bank ID and case ID are not used by
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

import vehicle_true_variable_horizon_history_feature_refit_v13 as v13  # noqa:E402

NAME = "vehicle_true_variable_horizon_calibrated_risk_value_refit_v14"
STAMP = "20260929T2215Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / "research_artifacts/aws_state/continue_state_20260929T2215_after_calibrated_risk_value_refit_v14.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-true-variable-H-calibrated-risk-value-refit-v14-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

FAMILIES = ["base_no_history", "history_no_risk", "history_with_risk"]


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


def row_key(row: Mapping[str, Any]) -> str:
    return v13.row_key(row)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def quantile(xs: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return 0.0
    # One-sided conservative empirical quantile.
    idx = max(0, min(len(vals) - 1, int(math.ceil(q * len(vals))) - 1))
    return vals[idx]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)) / max(1, len(a)))


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


def cfg_grid() -> List[Dict[str, Any]]:
    """Predeclared bounded grid for calibrated risk/value selectors.

    Count = 3 families * 3 k * 3 calibration quantiles * 4 risk thresholds
            * 2 physical-UCB thresholds * 3 gain-LCB thresholds * 2 support radii
          = 1296 configs.  With global LOBO plus nested LOBO this is 32,405
    split-level selector evaluations, below the v14 cap of 50,000.
    """
    cfgs: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for k in [1, 3, 5]:
            for calib_q in [0.50, 0.75, 0.90]:
                for risk_ucb_max in [0.25, 0.40, 0.60, 0.80]:
                    for phys_ucb_max in [2.0, 6.0]:
                        for gain_lcb_min in [0.0, 0.25, 0.50]:
                            for support_mult in [0.75, 1.25]:
                                cfgs.append({
                                    "family": family,
                                    "k": k,
                                    "calib_q": calib_q,
                                    "risk_ucb_max": risk_ucb_max,
                                    "phys_ucb_max": phys_ucb_max,
                                    "gain_lcb_min": gain_lcb_min,
                                    "support_mult": support_mult,
                                    "risk_alpha": 0.5,
                                })
    return cfgs


def cfg_id(c: Mapping[str, Any]) -> str:
    return "v14_%s_k%s_q%s_ru%s_pu%s_gl%s_sm%s" % (
        c["family"], c["k"], c["calib_q"], c["risk_ucb_max"], c["phys_ucb_max"], c["gain_lcb_min"], c["support_mult"]
    )


class ContextCache:
    def __init__(self, rows: Sequence[Mapping[str, Any]], all_features: Mapping[str, Mapping[str, float]], banks: Sequence[str]):
        self.rows = [dict(r) for r in rows]
        self.banks = list(banks)
        self.by_bank: Dict[str, List[Dict[str, Any]]] = {b: [dict(r) for r in self.rows if str(r["bank_id"]) == b] for b in banks}
        self.family_features: Dict[str, Dict[str, Dict[str, float]]] = {
            fam: v13.make_family_features(all_features, fam) for fam in FAMILIES
        }
        self.contexts: Dict[Tuple[str, Tuple[str, ...], str], Dict[str, Any]] = {}
        self.calibration: Dict[Tuple[str, Tuple[str, ...], int, float], Dict[str, float]] = {}
        self.feature_counts: Dict[str, int] = {}
        for fam in FAMILIES:
            self.feature_counts[fam] = len(v13.train_scaler(self.rows, self.family_features[fam])[0])

    def get(self, family: str, train_banks: Sequence[str], eval_bank: str) -> Dict[str, Any]:
        tb = tuple(sorted(str(b) for b in train_banks))
        key = (family, tb, str(eval_bank))
        if key in self.contexts:
            return self.contexts[key]
        train_set = set(tb)
        train_rows = [dict(r) for r in self.rows if str(r["bank_id"]) in train_set]
        eval_rows = [dict(r) for r in self.rows if str(r["bank_id"]) == str(eval_bank)]
        if not train_rows or not eval_rows:
            raise ContractError(f"empty context family={family} train={tb} eval={eval_bank}")
        fdict = self.family_features[family]
        names, means, stds = v13.train_scaler(train_rows, fdict)
        train_vecs = {row_key(r): v13.vec(r, fdict, names, means, stds) for r in train_rows}
        eval_vecs = {row_key(r): v13.vec(r, fdict, names, means, stds) for r in eval_rows}
        pos_train = [r for r in train_rows if bool(r["h10_beneficial_vs_h15"])]
        cat_train = [r for r in train_rows if bool(r["h10_catastrophic_vs_h15"])]
        pp: List[float] = []
        for i, r in enumerate(pos_train):
            ds = [dist(train_vecs[row_key(r)], train_vecs[row_key(q)]) for j, q in enumerate(pos_train) if j != i]
            if ds:
                pp.append(min(ds))
        pp.sort()
        neigh: Dict[str, Dict[str, List[Tuple[float, Dict[str, Any]]]]] = {}
        for er in eval_rows:
            ex = eval_vecs[row_key(er)]
            alln = sorted([(dist(ex, train_vecs[row_key(r)]), r) for r in train_rows], key=lambda z: (z[0], row_key(z[1])))
            posn = [(d, r) for d, r in alln if bool(r["h10_beneficial_vs_h15"])]
            catn = [(d, r) for d, r in alln if bool(r["h10_catastrophic_vs_h15"])]
            neigh[row_key(er)] = {"all": alln, "pos": posn, "cat": catn}
        ctx = {
            "family": family,
            "train_banks": list(tb),
            "eval_bank": str(eval_bank),
            "train_rows": train_rows,
            "eval_rows": eval_rows,
            "feature_count": len(names),
            "pp": pp,
            "neighbors": neigh,
        }
        self.contexts[key] = ctx
        return ctx

    def calibration_residuals(self, family: str, train_banks: Sequence[str], k: int, calib_q: float) -> Dict[str, float]:
        tb = tuple(sorted(str(b) for b in train_banks))
        key = (family, tb, int(k), float(calib_q))
        if key in self.calibration:
            return self.calibration[key]
        risk_resid: List[float] = []
        phys_resid: List[float] = []
        gain_resid: List[float] = []
        for cal_bank in tb:
            subtrain = [b for b in tb if b != cal_bank]
            if not subtrain:
                continue
            ctx = self.get(family, subtrain, cal_bank)
            for r in ctx["eval_rows"]:
                est = base_estimate(ctx, row_key(r), k)
                y_cat = 1.0 if bool(r["h10_catastrophic_vs_h15"]) else 0.0
                risk_resid.append(max(0.0, y_cat - sf(est["risk_hat"])))
                phys_resid.append(max(0.0, sf(r["phys_delta_h10_minus_h15"]) - sf(est["pred_phys"])))
                gain_resid.append(max(0.0, sf(est["pred_gain"]) - sf(r["decision_gain_h10_vs_h15_s"])))
        out = {
            "risk_q": quantile(risk_resid, calib_q),
            "phys_q": quantile(phys_resid, calib_q),
            "gain_q": quantile(gain_resid, calib_q),
            "n_calibration": len(risk_resid),
            "risk_resid_mean": mean(risk_resid),
            "phys_resid_mean": mean(phys_resid),
            "gain_resid_mean": mean(gain_resid),
        }
        self.calibration[key] = out
        return out


def support_radius(ctx: Mapping[str, Any], support_mult: float) -> float:
    pp = list(ctx.get("pp") or [])
    if not pp:
        return 1e9
    # Median positive-support radius, inflated by predeclared multiplier.
    idx = max(0, min(len(pp) - 1, int(math.floor(0.50 * (len(pp) - 1)))))
    return sf(pp[idx]) * support_mult


def base_estimate(ctx: Mapping[str, Any], key: str, k: int) -> Dict[str, float]:
    nb = ctx["neighbors"].get(key)
    if not nb or not nb.get("all"):
        return {"risk_hat": 1.0, "pred_gain": -1e9, "pred_phys": 1e9, "dpos": 1e9, "dcat": 0.0, "n_used": 0.0}
    alln = list(nb["all"])
    kk = max(1, min(int(k), len(alln)))
    near = alln[:kk]
    # Distance weights are capped by the standardized-distance scale.  The risk
    # estimate is smoothed to avoid declaring zero risk from one tiny neighborhood.
    weights = [1.0 / max(0.05, sf(d)) for d, _ in near]
    sw = sum(weights) if weights else 1.0
    cat_w = sum(w for w, (_, r) in zip(weights, near) if bool(r["h10_catastrophic_vs_h15"]))
    alpha = 0.5
    risk_hat = (cat_w + alpha) / (sw + 2.0 * alpha)
    pred_gain = sum(w * sf(r["decision_gain_h10_vs_h15_s"]) for w, (_, r) in zip(weights, near)) / sw
    pred_phys = sum(w * sf(r["phys_delta_h10_minus_h15"]) for w, (_, r) in zip(weights, near)) / sw
    dpos = sf(nb["pos"][0][0], 1e9) if nb.get("pos") else 1e9
    dcat = sf(nb["cat"][0][0], 1e9) if nb.get("cat") else 1e9
    return {"risk_hat": risk_hat, "pred_gain": pred_gain, "pred_phys": pred_phys, "dpos": dpos, "dcat": dcat, "n_used": kk}


def prediction_bounds(cache: ContextCache, family: str, train_banks: Sequence[str], eval_bank: str, row: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    key = row_key(row)
    k = si(cfg["k"], 3)
    calib = cache.calibration_residuals(family, train_banks, k, sf(cfg["calib_q"], 0.75))
    contexts = [cache.get(family, train_banks, eval_bank)]
    # Bank-jackknife uncertainty: if dropping one training bank materially changes
    # the local risk/value estimate, the max/min envelope drives abstention.
    for drop in sorted(set(str(b) for b in train_banks)):
        sub = [b for b in train_banks if b != drop]
        if sub:
            contexts.append(cache.get(family, sub, eval_bank))
    ests = [base_estimate(ctx, key, k) for ctx in contexts]
    full_radius = support_radius(contexts[0], sf(cfg["support_mult"], 1.0))
    support_ok = all(sf(e["dpos"], 1e9) <= support_radius(ctx, sf(cfg["support_mult"], 1.0)) for e, ctx in zip(ests, contexts))
    risk_raw = max(sf(e["risk_hat"], 1.0) for e in ests)
    phys_raw = max(sf(e["pred_phys"], 1e9) for e in ests)
    gain_raw = min(sf(e["pred_gain"], -1e9) for e in ests)
    risk_ucb = min(1.0, risk_raw + sf(calib["risk_q"]))
    phys_ucb = phys_raw + sf(calib["phys_q"])
    gain_lcb = gain_raw - sf(calib["gain_q"])
    choose = bool(
        support_ok
        and risk_ucb <= sf(cfg["risk_ucb_max"])
        and phys_ucb <= sf(cfg["phys_ucb_max"])
        and gain_lcb >= sf(cfg["gain_lcb_min"])
    )
    return {
        "selected_h": 10 if choose else 15,
        "risk_raw": risk_raw,
        "risk_ucb": risk_ucb,
        "phys_raw": phys_raw,
        "phys_ucb": phys_ucb,
        "gain_raw": gain_raw,
        "gain_lcb": gain_lcb,
        "support_ok": support_ok,
        "support_radius_full": full_radius,
        "dpos_full": sf(ests[0]["dpos"], 1e9),
        "dcat_full": sf(ests[0]["dcat"], 1e9),
        "calibration": calib,
    }


def eval_split(cache: ContextCache, cfg: Mapping[str, Any], train_banks: Sequence[str], eval_bank: str, include_scores: bool = False) -> Dict[str, Any]:
    family = str(cfg["family"])
    ctx = cache.get(family, train_banks, eval_bank)
    choices: Dict[str, int] = {}
    scores: Dict[str, Any] = {}
    for row in ctx["eval_rows"]:
        pred = prediction_bounds(cache, family, train_banks, eval_bank, row, cfg)
        choices[row_key(row)] = si(pred["selected_h"], 15)
        if include_scores:
            scores[row_key(row)] = pred
    ev = v13.v6.eval_choices(ctx["eval_rows"], choices)
    ev["feature_count"] = ctx["feature_count"]
    if include_scores:
        ev["prediction_scores"] = scores
    return ev


def aggregate_eval(rows: Sequence[Mapping[str, Any]], choices: Mapping[str, int]) -> Dict[str, Any]:
    return v13.v6.eval_choices(rows, choices)


def eval_lobo(cache: ContextCache, rows: Sequence[Mapping[str, Any]], banks: Sequence[str], cfg: Mapping[str, Any], blocked: Sequence[str] = (), include_scores: bool = False) -> Dict[str, Any]:
    blocked_set = set(str(b) for b in blocked)
    eval_banks = [b for b in banks if b not in blocked_set]
    holdouts: Dict[str, Any] = {}
    choices: Dict[str, int] = {}
    for hb in eval_banks:
        trb = [b for b in banks if b not in blocked_set and b != hb]
        ev = eval_split(cache, cfg, trb, hb, include_scores=include_scores)
        holdouts[hb] = ev
        for d in ev.get("details") or []:
            choices[str(d["bank_id"]) + "/" + str(d["base_state_id"])] = si(d.get("selected_h"), 15)
    eval_rows = [r for r in rows if str(r["bank_id"]) in eval_banks]
    agg = aggregate_eval(eval_rows, choices)
    return {"holdouts": holdouts, "aggregate": agg}


def summary_from_lobo(lobo: Mapping[str, Any]) -> Dict[str, Any]:
    agg = lobo["aggregate"]
    holdouts = lobo.get("holdouts") or {}
    hold_saves = [sf(e.get("decision_relative_saving_vs_fixed_H15")) for e in holdouts.values()]
    return {
        "aggregate_bad": len(agg.get("catastrophic_false_positive_rows") or []),
        "aggregate_physical_gate": bool(agg.get("physical_gate")),
        "aggregate_save": sf(agg.get("decision_relative_saving_vs_fixed_H15")),
        "aggregate_solver_save": sf(agg.get("solver_relative_saving_vs_fixed_H15")),
        "aggregate_h10": int((agg.get("chosen_counts") or {}).get("10", 0)),
        "aggregate_pass5": bool(agg.get("pass_5pct_no_cat_fp")),
        "aggregate_pass10": bool(agg.get("pass_10pct_no_cat_fp")),
        "min_holdout_save": min(hold_saves) if hold_saves else 0.0,
        "avg_holdout_save": mean(hold_saves),
        "holdout_bad_total": sum(len(e.get("catastrophic_false_positive_rows") or []) for e in holdouts.values()),
        "holdout_phys_fail_count": sum(0 if e.get("physical_gate") else 1 for e in holdouts.values()),
    }


def rank_key(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        int(s["aggregate_bad"]),
        0 if bool(s["aggregate_physical_gate"]) else 1,
        not bool(s["aggregate_pass10"]),
        not bool(s["aggregate_pass5"]),
        -sf(s["aggregate_save"]),
        -sf(s["aggregate_h10"]),
        int(s["holdout_bad_total"]),
        int(s["holdout_phys_fail_count"]),
    )


def compact_lobo(lobo: Mapping[str, Any]) -> Dict[str, Any]:
    out = {"aggregate": lobo.get("aggregate"), "holdouts": {}}
    for b, ev in (lobo.get("holdouts") or {}).items():
        keep = dict(ev)
        # Keep split details, but prediction score dictionaries can be very large;
        # callers decide when to include them.
        out["holdouts"][b] = keep
    return out


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    lines: List[str] = [
        "# Vehicle true-variable-H v14 calibrated risk/value refit",
        "",
        f"UTC `{raw['created_utc']}`. Development-only IMPROVED calibrated risk/value refit over opened banks; no MPC simulation, no validation64, no sealed test, no gradient/RL training.",
        "",
        "## Headline",
        "",
        f"- Rows `{h['rows']}` banks `{h['banks']}`; positives `{h['positive_rows']}`; catastrophic H10 rows `{h['catastrophic_rows']}`; H15-prefix traces `{h['trace_loaded_rows']}/{h['rows']}`.",
        f"- Feature counts `{h['feature_counts_by_family']}`; configs `{h['config_count']}`; split-level selector evaluations `{raw['budget_actual']['selector_refit_evaluations']}`.",
        f"- Global aggregate pass10 `{h['global_pass10_count']}`, pass5 `{h['global_pass5_count']}`; best global `{h['best_global_config']}` save `{pct(h['best_global_save'])}`, bad `{h['best_global_bad']}`, H10 `{h['best_global_h10']}`.",
        f"- Nested aggregate save `{pct(h['nested_save'])}`, solver save `{pct(h['nested_solver_save'])}`, bad `{h['nested_bad']}`, physical gate `{h['nested_physical_gate']}`, H10 `{h['nested_h10']}`, pass5 `{h['nested_pass5']}`, pass10 `{h['nested_pass10']}`.",
        f"- Decision: {raw['decision']}",
        "",
        "## Nested outer-bank calibrated selection",
        "",
        "| outer bank | selected config | inner pass5 | inner bad | inner save | outer H counts | outer bad | outer physical gate | outer save |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b, obj in raw["nested_outer"].items():
        ev = obj["outer_eval"]
        inn = obj["inner_summary"]
        lines.append(
            f"| `{b}` | `{obj['selected_config_id']}` | `{inn['aggregate_pass5']}` | {inn['aggregate_bad']} | {pct(inn['aggregate_save'])} | `{ev['chosen_counts']}` | {len(ev['catastrophic_false_positive_rows'])} | `{ev['physical_gate']}` | {pct(ev['decision_relative_saving_vs_fixed_H15'])} |"
        )
    lines += ["", "## Top global LOBO aggregate configs", "", "| rank | config | pass10 | pass5 | bad | physical gate | save | solver save | H10 | min holdout save |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, x in enumerate(raw["top_global_lobo"][:15], 1):
        s = x["summary"]
        lines.append(
            f"| {i} | `{x['config_id']}` | `{s['aggregate_pass10']}` | `{s['aggregate_pass5']}` | {s['aggregate_bad']} | `{s['aggregate_physical_gate']}` | {pct(s['aggregate_save'])} | {pct(s['aggregate_solver_save'])} | {s['aggregate_h10']} | {pct(s['min_holdout_save'])} |"
        )
    lines += ["", "## Nested catastrophic false positives", ""]
    fps = raw.get("nested_aggregate", {}).get("catastrophic_false_positive_rows") or []
    if fps:
        lines.append("| bank | state | group | window | phys delta | tol | decision gain s |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for r in fps:
            lines.append(f"| `{r.get('bank_id')}` | `{r.get('base_state_id')}` | `{r.get('group')}` | `{r.get('window')}` | {sf(r.get('phys_delta')):.6g} | {sf(r.get('tol')):.6g} | {sf(r.get('decision_gain_s')):.6g} |")
    else:
        lines.append("No nested catastrophic false positives.")
    lines += [
        "",
        "## Interpretation",
        "",
        "This remains opened-development evidence only. A pass would justify only an unused-source overhead-aware confirmation with actual selector overhead before any validation64. A failure indicates that calibrated uncertainty over the current deployable representation/data still cannot extract the v12 oracle opportunity safely, strengthening the case for targeted boundary acquisition or a different terminal/risk-value learning intervention.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    elapsed_h = (now() - FIRST_EVENT).total_seconds() / 3600.0
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v14 calibrated risk/value refit

Elapsed service lifetime at write: >{elapsed_h:.1f} h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED calibrated risk/value refit over opened banks; no MPC simulation, validation64 or sealed test. configs={h['config_count']}; selector_refit_evaluations={raw['budget_actual']['selector_refit_evaluations']}; global_pass5={h['global_pass5_count']}; nested_save={h['nested_save']:.4f}; nested_bad={h['nested_bad']}; nested_pass5={h['nested_pass5']}; decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
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
            f.write(f"{STAMP},{NAME},development_calibrated_risk_value_refit,opened_banks_nested_lobo_no_sim_no_validation_no_test,0,0,{raw['budget_actual']['selector_refit_evaluations']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    created = now()
    rows, input_hashes = v13.load_rows()
    all_features, hist_diag, hist_hashes = v13.enrich_history(rows)
    input_hashes.update(hist_hashes)
    banks = sorted(set(str(r["bank_id"]) for r in rows))
    if banks != ["fresh_v0", "fresh_v1", "fresh_v11", "fresh_v2", "fresh_v8c"] or len(rows) != 68:
        raise ContractError(f"unexpected opened rows/banks {len(rows)} {banks}")
    for r in rows:
        bid = str(r.get("bank_id", "")).lower()
        if "validation" in bid or "test" in bid:
            raise ContractError("forbidden validation/test-looking bank id: " + str(r.get("bank_id")))
    cfgs = cfg_grid()
    declared_eval_calls = len(cfgs) * len(banks) + len(banks) * (len(cfgs) * (len(banks) - 1) + 1)
    if declared_eval_calls > 50000:
        raise ContractError(f"declared selector-evaluation cap exceeded: {declared_eval_calls}")
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_calibrated_risk_value_refit_no_simulation_no_validation_no_test",
        "hypothesis": "A deployable calibrated risk/value selector using leave-bank-calibrated residuals and bank-jackknife uncertainty can abstain on ambiguous H10 states while preserving at least 5% aggregate measured branch decision-time saving over fixed H15 on opened-development nested LOBO splits with zero catastrophic H10 false positives.",
        "before_evidence": [
            "v12 aligned oracle over opened banks saved 30.8% branch decision time with physical gate, establishing adaptive opportunity under outcome-aligned labels.",
            "v13b loaded all 68 H15-prefix traces but nested deployment saved only 1.5% and still produced one catastrophic H10 false positive; no global or nested pass5 config existed.",
            "v13b feature-space ambiguity persisted after H15-prefix history, suggesting uncertainty/risk calibration or more boundary labels rather than another deterministic support threshold sweep."
        ],
        "split": "nested leave-opened-development-bank-out over fresh_v0/fresh_v1/fresh_v2/fresh_v8c/fresh_v11 only; validation64 and sealed test forbidden",
        "features": "deployable instantaneous observation/pose/progress and optional H15-prefix history features from v13; no bank ID, case ID, scenario group/window, outcome label, validation or test data as candidate inputs",
        "model": "distance-kernel H10 catastrophic-risk and H10-vs-H15 gain/physical-delta estimators with leave-bank calibration residuals, bank-jackknife uncertainty envelope and support abstention to H15",
        "grid": {
            "family": FAMILIES,
            "k": [1, 3, 5],
            "calib_q": [0.50, 0.75, 0.90],
            "risk_ucb_max": [0.25, 0.40, 0.60, 0.80],
            "phys_ucb_max": [2.0, 6.0],
            "gain_lcb_min": [0.0, 0.25, 0.50],
            "support_mult": [0.75, 1.25],
            "risk_alpha": [0.5]
        },
        "config_count": len(cfgs),
        "model_selection_rule": "For global diagnostics and nested inner selection, rank by aggregate catastrophic false positives, aggregate physical gate, pass10/pass5 status, aggregate decision saving, then H10 count; outer aggregate is evaluated once after inner selection.",
        "decision_gate": "weak pass if nested aggregate has zero catastrophic H10 false positives, physical gate true, and >=5% measured branch decision saving vs fixed H15; strong pass if >=10%; otherwise no validation64 rollout",
        "budget_declared": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "training_episodes": 0, "gradient_steps": 0, "selector_refit_evaluations_max": 50000, "selector_refit_evaluations_declared": declared_eval_calls, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "input_hashes": {**input_hashes, rel(SOURCE): sha256(SOURCE)},
        "history_availability_preoutcome": hist_diag,
    }
    write_json(PROTOCOL, protocol)

    cache = ContextCache(rows, all_features, banks)
    selector_evals = 0
    global_candidates: List[Dict[str, Any]] = []
    for cfg in cfgs:
        lobo = eval_lobo(cache, rows, banks, cfg, include_scores=False)
        selector_evals += len(banks)
        summ = summary_from_lobo(lobo)
        global_candidates.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summ})
    ranked = sorted(global_candidates, key=lambda x: rank_key(x["summary"]))
    top_global: List[Dict[str, Any]] = []
    for x in ranked[:40]:
        lobo = eval_lobo(cache, rows, banks, x["config"], include_scores=True)
        top_global.append({"config_id": x["config_id"], "config": x["config"], "summary": x["summary"], "lobo": compact_lobo(lobo)})

    nested_outer: Dict[str, Any] = {}
    nested_choices: Dict[str, int] = {}
    for outer_bank in banks:
        inner_candidates: List[Dict[str, Any]] = []
        inner_banks = [b for b in banks if b != outer_bank]
        inner_rows = [r for r in rows if str(r["bank_id"]) != outer_bank]
        for cfg in cfgs:
            lobo = eval_lobo(cache, rows, banks, cfg, blocked=[outer_bank], include_scores=False)
            selector_evals += len(inner_banks)
            summ = summary_from_lobo(lobo)
            inner_candidates.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summ})
        selected = sorted(inner_candidates, key=lambda x: rank_key(x["summary"]))[0]
        outer_eval = eval_split(cache, selected["config"], inner_banks, outer_bank, include_scores=True)
        selector_evals += 1
        for d in outer_eval.get("details") or []:
            nested_choices[str(d["bank_id"]) + "/" + str(d["base_state_id"])] = si(d.get("selected_h"), 15)
        nested_outer[outer_bank] = {
            "selected_config_id": selected["config_id"],
            "selected_config": selected["config"],
            "inner_summary": selected["summary"],
            "inner_rows": len(inner_rows),
            "outer_eval": outer_eval,
        }
    nested_agg = aggregate_eval(rows, nested_choices)

    headline = {
        "rows": len(rows),
        "banks": banks,
        "positive_rows": sum(1 for r in rows if bool(r["h10_beneficial_vs_h15"])),
        "catastrophic_rows": sum(1 for r in rows if bool(r["h10_catastrophic_vs_h15"])),
        "trace_loaded_rows": hist_diag.get("trace_loaded_rows"),
        "feature_counts_by_family": cache.feature_counts,
        "config_count": len(cfgs),
        "declared_selector_refit_evaluations": declared_eval_calls,
        "global_pass10_count": sum(1 for x in global_candidates if bool(x["summary"]["aggregate_pass10"])),
        "global_pass5_count": sum(1 for x in global_candidates if bool(x["summary"]["aggregate_pass5"])),
        "best_global_config": ranked[0]["config_id"],
        "best_global_save": ranked[0]["summary"]["aggregate_save"],
        "best_global_solver_save": ranked[0]["summary"]["aggregate_solver_save"],
        "best_global_bad": ranked[0]["summary"]["aggregate_bad"],
        "best_global_h10": ranked[0]["summary"]["aggregate_h10"],
        "nested_save": sf(nested_agg.get("decision_relative_saving_vs_fixed_H15")),
        "nested_solver_save": sf(nested_agg.get("solver_relative_saving_vs_fixed_H15")),
        "nested_bad": len(nested_agg.get("catastrophic_false_positive_rows") or []),
        "nested_physical_gate": bool(nested_agg.get("physical_gate")),
        "nested_h10": int((nested_agg.get("chosen_counts") or {}).get("10", 0)),
        "nested_pass5": bool(nested_agg.get("pass_5pct_no_cat_fp")),
        "nested_pass10": bool(nested_agg.get("pass_10pct_no_cat_fp")),
        "nested_selected_family_counts": dict(Counter(str(x["selected_config"].get("family")) for x in nested_outer.values())),
    }
    if headline["nested_pass10"]:
        decision = "v14 passes strong opened-development calibrated risk/value gate; next freeze a small unused-source overhead-aware confirmation with actual selector overhead, still no validation64/sealed test."
    elif headline["nested_pass5"]:
        decision = "v14 passes weak opened-development calibrated risk/value gate; next freeze unused-source overhead-aware confirmation requiring zero catastrophic H10 before any validation64."
    elif headline["global_pass5_count"] > 0:
        decision = "Some global calibrated risk/value configs pass aggregate opened-bank gates, but nested model selection does not; next inspect selected outer failures and acquire targeted boundary/calibration labels before validation rollout."
    else:
        decision = "Calibrated uncertainty over the current deployable feature/data representation still cannot safely extract >=5% opened-bank savings; next run targeted boundary acquisition or a richer terminal/risk-value learning refit rather than another static/history feature sweep or validation rollout."

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
        "top_global_lobo": top_global,
        "nested_outer": nested_outer,
        "nested_aggregate": nested_agg,
        "decision": decision,
        "input_hashes": {**input_hashes, rel(PROTOCOL): sha256(PROTOCOL), rel(SOURCE): sha256(SOURCE)},
        "platform": {"python": sys.version, "platform": platform.platform()},
    }
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-calibrated-risk-value-v14", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_calibrated_risk_value_v14:
        raise ContractError("requires --run and explicit v14 development acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_CALIBRATED_RISK_VALUE_REFIT_V14_{STAMP}.json"
    write_json(req, {"created_utc": raw["created_utc"], "reason": "backup after v14 calibrated risk/value refit before any further science", "paths": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(SOURCE), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "validation64_bank_opened": False, "sealed_test_accessed": False})
    done = {
        "passed": True,
        "status": "complete",
        "marker": MARKER,
        "summary": rel(OUT / "summary.md"),
        "raw": rel(OUT / "raw.json"),
        "protocol": rel(PROTOCOL),
        "backup_request": rel(req),
        "headline": raw["headline"],
        "decision": raw["decision"],
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_mpc_simulation_episodes": 0,
        "new_control_steps": 0,
        "training_episodes": 0,
        "gradient_steps": 0,
        "selector_refit_evaluations": raw["budget_actual"]["selector_refit_evaluations"],
        "hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "raw.json"): sha256(OUT / "raw.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md"), rel(req): sha256(req)},
    }
    write_json(OUT / "completed.json", done)
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Continue state after v14 calibrated risk/value refit\n\n" + json.dumps(clean({"utc": raw["created_utc"], "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "completed": rel(OUT / "completed.json"), "protocol": rel(PROTOCOL), "backup_request": rel(req)}, "current_backup_status": "not verified after v14; run backup before further scientific simulation/refit", "next_action": "Follow v14 decision. If pass, freeze unused-source overhead-aware confirmation; otherwise run targeted boundary acquisition or richer terminal/risk-value learning. No validation64/sealed test."}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
