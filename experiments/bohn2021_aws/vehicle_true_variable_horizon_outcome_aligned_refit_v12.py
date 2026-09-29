#!/usr/bin/env python3
"""v12 outcome-aligned risk/value selector refit for true variable H10/H15.

Development-only IMPROVED diagnostic after v11.  v11 exposed a likely objective
/ label mismatch: rows where fixed H15 failed or was vastly worse were not marked
H10-beneficial by the earlier label definition, so the old "oracle" could choose
H15 on states where H10 was safer and much faster.  This script freezes and
executes a bounded relabel/refit over opened development banks only.

Primary hypothesis: if the bottleneck is partly the old label/objective mismatch,
then an outcome-aligned label that treats safe H10 as beneficial when H15 is
unsafe (or when H10 is within the physical tolerance and faster) will improve the
nested leave-opened-bank-out deployable selector tradeoff without increasing
catastrophic H10 false positives.  If it still fails, the next intervention
should be richer risk/terminal-value training or targeted data acquisition, not
another unchanged selector sweep.

No MPC simulation, no validation64 access, no sealed test access, and no gradient
training are performed.
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
import vehicle_true_variable_horizon_risk_refit_v9_opened_banks as v9  # noqa:E402

NAME = "vehicle_true_variable_horizon_outcome_aligned_refit_v12"
STAMP = "20260929T1855Z"
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
PROTOCOL = ROOT / f"research_artifacts/aws_protocols/{NAME}_preoutcome_frozen_{STAMP}.json"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_20260929T1855_after_outcome_aligned_refit_v12.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
SOURCE = Path(__file__).resolve()
MARKER = f"vehicle-true-variable-H-outcome-aligned-refit-v12-{STAMP}"
FIRST_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")

V11_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z"
V11_RAW = V11_DIR / "raw.json"
V11_DONE = V11_DIR / "completed.json"
V11_MANIFEST = V11_DIR / "selected_state_manifest.json"
V11_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_preoutcome_frozen_20260929T1835Z.json"
V10B_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/completed.json"
V10B_SUMMARY = ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/summary.md"

MIN_ROW_TOL = 2.0

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


def obs14(st: Mapping[str, Any]) -> List[float]:
    raw = st.get("initial_observation_from_h15_trace") or st.get("initial_observation_at_branch") or []
    vals = [sf(v) for v in raw[:14]] if isinstance(raw, list) else []
    while len(vals) < 14:
        vals.append(0.0)
    return vals[:14]


def load_v11_rows() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    for p in (V11_RAW, V11_DONE, V11_MANIFEST, V11_PROTOCOL):
        if not p.exists():
            raise ContractError("missing v11 input: " + rel(p))
    raw = read_json(V11_RAW)
    done = read_json(V11_DONE)
    man = read_json(V11_MANIFEST)
    assert_dev_only(raw, "v11 raw")
    assert_dev_only(done, "v11 completed")
    assert_dev_only(man, "v11 manifest")
    if done.get("hard_pass") is not True and done.get("passed") is not True and done.get("status") not in ("complete", "completed"):
        raise ContractError("v11 prerequisite did not complete")
    by_base = {str(s.get("base_state_id")): s for s in (man.get("selected_states") or [])}
    rows: List[Dict[str, Any]] = []
    for r in ((raw.get("analysis") or {}).get("state_rows") or []):
        sid = str(r.get("base_state_id"))
        st = by_base.get(sid, {})
        h10 = r.get("h10") or {}
        h15 = r.get("h15") or {}
        prev = st.get("branch_previous_state") if isinstance(st.get("branch_previous_state"), Mapping) else {}
        role = str(r.get("risk_probe_role") or st.get("risk_probe_role") or st.get("fresh_confirmation_group") or "v11_unknown")
        window = str(r.get("slot_name") or st.get("window") or "unknown")
        h10_phys = sf(h10.get("physical")); h15_phys = sf(h15.get("physical"))
        h10_dec = sf(h10.get("decision_sum_s")); h15_dec = sf(h15.get("decision_sum_s"))
        h10_sol = sf(h10.get("solver_sum_s")); h15_sol = sf(h15.get("solver_sum_s"))
        rows.append({
            "bank_id": "fresh_v11",
            "base_state_id": sid,
            "group": role,
            "window": window,
            "branch_state_slot": si(r.get("branch_state_slot", st.get("branch_state_slot")), 0),
            "branch_step": si(r.get("branch_step", st.get("branch_step")), -1),
            "obs14": obs14(st),
            "prev_x": sf(prev.get("x")),
            "prev_y": sf(prev.get("y")),
            "prev_theta": sf(prev.get("theta")),
            "stage_a_trace_risk_score": sf(st.get("stage_a_trace_risk_score"), 0.0),
            "h10_physical": h10_phys,
            "h15_physical": h15_phys,
            "h10_decision_sum_s": h10_dec,
            "h15_decision_sum_s": h15_dec,
            "h10_solver_sum_s": h10_sol,
            "h15_solver_sum_s": h15_sol,
            "h10_safe_all": bool(h10.get("safe_all")),
            "h15_safe_all": bool(h15.get("safe_all")),
            "h10_success_all": bool(h10.get("success_all", h10.get("safe_all", False))),
            "h15_success_all": bool(h15.get("success_all", h15.get("safe_all", False))),
            "row_physical_tolerance": max(MIN_ROW_TOL, sf(r.get("row_tolerance_vs_H15"), MIN_ROW_TOL)),
            "phys_delta_h10_minus_h15": sf(r.get("physical_delta_h10_minus_h15", h10_phys - h15_phys)),
            "decision_gain_h10_vs_h15_s": sf(r.get("decision_delta_h10_minus_h15"), h15_dec - h10_dec) * -1.0 if "decision_delta_h10_minus_h15" in r else sf(h15_dec - h10_dec),
            "solver_gain_h10_vs_h15_s": h15_sol - h10_sol,
            "h10_beneficial_vs_h15": bool(r.get("h10_beneficial_vs_h15")),
            "h10_catastrophic_vs_h15": bool(r.get("h10_catastrophic_vs_h15")),
            "source_candidate_index": si(r.get("source_candidate_index"), -1),
            "risk_probe_role": role,
        })
    if len(rows) != 12:
        raise ContractError(f"expected 12 v11 rows, got {len(rows)}")
    return rows, {rel(V11_RAW): sha256(V11_RAW), rel(V11_DONE): sha256(V11_DONE), rel(V11_MANIFEST): sha256(V11_MANIFEST), rel(V11_PROTOCOL): sha256(V11_PROTOCOL)}


def load_all_rows() -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    rows0, hashes, groups0 = v6.load_rows()
    rows8, h8 = v9.load_v8c_rows()
    rows11, h11 = load_v11_rows()
    rows = [dict(r) for r in rows0 + rows8 + rows11]
    hashes.update(h8)
    hashes.update(h11)
    for p in (SOURCE, Path(v6.__file__), Path(v9.__file__), V10B_DONE, V10B_SUMMARY):
        if p.exists():
            hashes[rel(p)] = sha256(p)
    banks = sorted(set(str(r.get("bank_id")) for r in rows))
    if banks != ["fresh_v0", "fresh_v1", "fresh_v11", "fresh_v2", "fresh_v8c"] or len(rows) != 68:
        raise ContractError(f"unexpected opened development rows/banks: {len(rows)} {banks}")
    for r in rows:
        bid = str(r.get("bank_id", "")).lower()
        if "validation" in bid or "test" in bid:
            raise ContractError("forbidden validation/test-looking bank id: " + str(r.get("bank_id")))
    all_groups = set(groups0)
    for r in rows:
        all_groups.add(str(r.get("group")))
        all_groups.add(f"window::{r.get('window')}")
        all_groups.add(f"gw::{r.get('group')}::{r.get('window')}")
    return rows, hashes, sorted(all_groups)


def aligned_label(row: Mapping[str, Any]) -> Tuple[bool, bool, str]:
    h10_safe = bool(row.get("h10_safe_all")) and bool(row.get("h10_success_all", True))
    h15_safe = bool(row.get("h15_safe_all")) and bool(row.get("h15_success_all", True))
    phys_delta = sf(row.get("phys_delta_h10_minus_h15"))
    tol = max(MIN_ROW_TOL, sf(row.get("row_physical_tolerance"), MIN_ROW_TOL))
    gain = sf(row.get("decision_gain_h10_vs_h15_s"))
    if not h10_safe:
        return False, True, "h10_unsafe_or_unsuccessful"
    if h15_safe:
        if phys_delta > tol:
            return False, True, "h15_safe_h10_exceeds_tolerance"
        if gain > 0:
            return True, False, "h15_safe_h10_within_tolerance_and_faster"
        return False, False, "h15_safe_h10_not_faster"
    # H15 is unsafe/unsuccessful: safe H10 is desirable when it is not slower.
    if gain > 0:
        return True, False, "h15_unsafe_safe_h10_faster"
    return False, False, "h15_unsafe_safe_h10_not_faster"


def relabel_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    flips: List[Dict[str, Any]] = []
    reason_counts: Counter = Counter()
    for row in rows:
        r = dict(row)
        r["old_h10_beneficial_vs_h15"] = bool(r.get("h10_beneficial_vs_h15"))
        r["old_h10_catastrophic_vs_h15"] = bool(r.get("h10_catastrophic_vs_h15"))
        pos, cat, reason = aligned_label(r)
        r["h10_beneficial_vs_h15"] = pos
        r["h10_catastrophic_vs_h15"] = cat
        r["outcome_aligned_label_reason"] = reason
        reason_counts[reason] += 1
        if pos != r["old_h10_beneficial_vs_h15"] or cat != r["old_h10_catastrophic_vs_h15"]:
            flips.append({
                "bank_id": r.get("bank_id"),
                "base_state_id": r.get("base_state_id"),
                "old_positive": r["old_h10_beneficial_vs_h15"],
                "new_positive": pos,
                "old_catastrophic": r["old_h10_catastrophic_vs_h15"],
                "new_catastrophic": cat,
                "reason": reason,
                "h10_safe": bool(r.get("h10_safe_all")),
                "h15_safe": bool(r.get("h15_safe_all")),
                "phys_delta": sf(r.get("phys_delta_h10_minus_h15")),
                "tol": max(MIN_ROW_TOL, sf(r.get("row_physical_tolerance"), MIN_ROW_TOL)),
                "decision_gain_s": sf(r.get("decision_gain_h10_vs_h15_s")),
                "group": r.get("group"),
                "window": r.get("window"),
            })
        out.append(r)
    return out, {"reason_counts": dict(reason_counts), "flipped_rows": flips, "flip_count": len(flips)}


def choices_from_label(rows: Sequence[Mapping[str, Any]], positive_field: str) -> Dict[str, int]:
    return {v6.row_key(r): (10 if bool(r.get(positive_field)) else 15) for r in rows}


def all_h_choices(rows: Sequence[Mapping[str, Any]], h: int) -> Dict[str, int]:
    return {v6.row_key(r): h for r in rows}


def cfg_grid() -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    families = [
        "deploy_obs_pose_step_no_risk",
        "deploy_obs_pose_step_risk",
        "step_window_only",
        "scenario_group_window_only",
        "deploy_plus_group_window",
    ]
    for fam in families:
        for depth in [1, 2, 3, 4]:
            for min_leaf in [2, 3, 4, 5]:
                for min_pos in [1, 2, 3]:
                    for min_gain in [0.0, 0.25, 0.5, 1.0, 1.5]:
                        cfgs.append({
                            "family": fam,
                            "max_depth": depth,
                            "min_leaf": min_leaf,
                            "min_pos_leaf": min_pos,
                            "max_train_cat_leaf": 0,
                            "min_leaf_gain_s": min_gain,
                        })
    return cfgs


def cfg_id(c: Mapping[str, Any]) -> str:
    return "v12_%s_d%s_ml%s_mp%s_mg%s" % (c["family"], c["max_depth"], c["min_leaf"], c["min_pos_leaf"], str(c["min_leaf_gain_s"]).replace(".", "p"))


def fit_eval(train: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], all_groups: Sequence[str], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    fdicts = {v6.row_key(r): v6.feature_dict(r, all_groups) for r in list(train) + list(eval_rows)}
    names = v6.feature_names(str(cfg["family"]), all_groups)
    tree = v6.build_tree(train, fdicts, names, int(cfg["max_depth"]), int(cfg["min_leaf"]), int(cfg["min_pos_leaf"]), int(cfg["max_train_cat_leaf"]), sf(cfg["min_leaf_gain_s"]))
    choices = {v6.row_key(r): v6.predict(tree, fdicts[v6.row_key(r)]) for r in eval_rows}
    ev = v6.eval_choices(eval_rows, choices)
    ev["tree"] = v6.compact_tree(tree)
    return ev


def eval_lobo(rows: Sequence[Mapping[str, Any]], banks: Sequence[str], all_groups: Sequence[str], cfg: Mapping[str, Any], blocked: Sequence[str] = ()) -> Tuple[Dict[str, Any], int]:
    blocked_set = set(blocked)
    hold: Dict[str, Any] = {}
    fits = 0
    for hb in banks:
        if hb in blocked_set:
            continue
        train = [r for r in rows if str(r["bank_id"]) not in blocked_set and str(r["bank_id"]) != hb]
        evrows = [r for r in rows if str(r["bank_id"]) == hb]
        hold[hb] = fit_eval(train, evrows, all_groups, cfg)
        fits += 1
    return hold, fits


def summarize(hold: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    bad = sum(len(e.get("catastrophic_false_positive_rows") or []) for e in hold.values())
    phys_fail = sum(0 if e.get("physical_gate") else 1 for e in hold.values())
    saves = [sf(e.get("decision_relative_saving_vs_fixed_H15"), -99.0) for e in hold.values()]
    h10 = sum(int((e.get("chosen_counts") or {}).get("10", 0)) for e in hold.values())
    return {
        "all_pass10": all(bool(e.get("pass_10pct_no_cat_fp")) for e in hold.values()),
        "all_pass5": all(bool(e.get("pass_5pct_no_cat_fp")) for e in hold.values()),
        "total_bad": bad,
        "phys_fail": phys_fail,
        "min_save": min(saves) if saves else 0.0,
        "avg_save": sum(saves) / len(saves) if saves else 0.0,
        "total_h10": h10,
    }


def rank_summary(s: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        int(s["total_bad"]),
        int(s["phys_fail"]),
        not bool(s["all_pass10"]),
        not bool(s["all_pass5"]),
        -sf(s["min_save"]),
        -sf(s["avg_save"]),
        -int(s["total_h10"]),
    )


def eval_nested(rows: Sequence[Mapping[str, Any]], banks: Sequence[str], all_groups: Sequence[str], cfgs: Sequence[Mapping[str, Any]], subset_name: str) -> Tuple[Dict[str, Any], int]:
    fits = 0
    outer: Dict[str, Any] = {}
    choices: Dict[str, int] = {}
    for outer_bank in banks:
        inner_candidates: List[Dict[str, Any]] = []
        for cfg in cfgs:
            hold, nf = eval_lobo(rows, banks, all_groups, cfg, blocked=[outer_bank])
            fits += nf
            summ = summarize(hold)
            inner_candidates.append({"config_id": cfg_id(cfg), "config": cfg, "inner_summary": summ})
        selected = sorted(inner_candidates, key=lambda x: rank_summary(x["inner_summary"]))[0]
        train_outer = [r for r in rows if str(r["bank_id"]) != outer_bank]
        eval_outer = [r for r in rows if str(r["bank_id"]) == outer_bank]
        outer_eval = fit_eval(train_outer, eval_outer, all_groups, selected["config"])
        fits += 1
        for d in outer_eval.get("details") or []:
            choices[str(d["bank_id"]) + "/" + str(d["base_state_id"])] = int(d["selected_h"])
        outer[outer_bank] = {
            "selected_config_id": selected["config_id"],
            "selected_config": selected["config"],
            "inner_summary": selected["inner_summary"],
            "outer_eval": outer_eval,
        }
    aggregate = v6.eval_choices(rows, choices)
    return {"subset": subset_name, "outer": outer, "aggregate": aggregate}, fits


def pct(x: Any) -> str:
    return f"{100.0 * sf(x):.2f}%"


def make_old_label_copy(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r["h10_beneficial_vs_h15"] = bool(r.get("old_h10_beneficial_vs_h15"))
        r["h10_catastrophic_vs_h15"] = bool(r.get("old_h10_catastrophic_vs_h15"))
        out.append(r)
    return out


def write_summary(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    label = raw["label_audit"]
    lines = [
        "# Vehicle true-variable-H v12 outcome-aligned refit",
        "",
        f"UTC `{raw['created_utc']}`. Development-only IMPROVED objective/label refit over opened banks; no MPC simulation, validation64, sealed test, or gradient training.",
        "",
        "## Motivation",
        "",
        "v11 revealed rows where H15 failed while H10 succeeded with far lower physical cost, but the previous positive-label/oracle definition still chose H15. v12 repairs this development objective mismatch by marking safe/faster H10 as positive when H15 is unsafe, while retaining a zero-catastrophic-H10 false-positive gate.",
        "",
        "## Headline",
        "",
        f"- Rows `{h['rows']}` across banks `{h['banks']}`; old positives `{h['old_positive_rows']}`, aligned positives `{h['aligned_positive_rows']}`; aligned catastrophic H10 rows `{h['aligned_catastrophic_rows']}`.",
        f"- Label flips `{label['flip_count']}`; reasons `{label['reason_counts']}`.",
        f"- Old oracle: save `{pct(h['old_oracle_save'])}`, physical gate `{h['old_oracle_physical_gate']}`, physical delta `{h['old_oracle_physical_delta']:.6g}`.",
        f"- Aligned oracle: save `{pct(h['aligned_oracle_save'])}`, physical gate `{h['aligned_oracle_physical_gate']}`, physical delta `{h['aligned_oracle_physical_delta']:.6g}`.",
        f"- Fixed H10 diagnostic: save `{pct(h['fixed_h10_save'])}`, physical gate `{h['fixed_h10_physical_gate']}`, catastrophic rows `{h['fixed_h10_bad']}`.",
        f"- Global pure-deploy pass10 `{h['global_deploy_pass10_count']}`, pass5 `{h['global_deploy_pass5_count']}`; all-family pass10 `{h['global_all_pass10_count']}`, pass5 `{h['global_all_pass5_count']}`.",
        f"- Nested deploy aggregate: save `{pct(h['nested_deploy_save'])}`, bad `{h['nested_deploy_bad']}`, physical gate `{h['nested_deploy_physical_gate']}`, H10 count `{h['nested_deploy_h10']}`, pass5 `{h['nested_deploy_pass5']}`, pass10 `{h['nested_deploy_pass10']}`.",
        f"- Nested all-family aggregate: save `{pct(h['nested_all_save'])}`, bad `{h['nested_all_bad']}`, physical gate `{h['nested_all_physical_gate']}`, H10 count `{h['nested_all_h10']}`, pass5 `{h['nested_all_pass5']}`, pass10 `{h['nested_all_pass10']}`.",
        f"- Decision: {raw['decision']}",
        "",
        "## Label flips (old objective -> outcome-aligned objective)",
        "",
        "| bank | state | old pos | new pos | old cat | new cat | reason | h10 safe | h15 safe | physΔ | tol | gain s | group | window |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for f in label["flipped_rows"][:50]:
        lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | %.6g | %.6g | %.6g | `%s` | `%s` |" % (
            f.get("bank_id"), f.get("base_state_id"), f.get("old_positive"), f.get("new_positive"), f.get("old_catastrophic"), f.get("new_catastrophic"), f.get("reason"), f.get("h10_safe"), f.get("h15_safe"), sf(f.get("phys_delta")), sf(f.get("tol")), sf(f.get("decision_gain_s")), f.get("group"), f.get("window")))
    lines += [
        "",
        "## Nested deployable outer-bank model selection",
        "",
        "| outer bank | selected config | inner pass5 | inner bad | outer H counts | outer bad | outer physical gate | outer save |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b, x in raw["nested_deploy"]["outer"].items():
        ev = x["outer_eval"]
        lines.append(f"| `{b}` | `{x['selected_config_id']}` | `{x['inner_summary']['all_pass5']}` | {x['inner_summary']['total_bad']} | `{ev['chosen_counts']}` | {len(ev['catastrophic_false_positive_rows'])} | `{ev['physical_gate']}` | {pct(ev['decision_relative_saving_vs_fixed_H15'])} |")
    lines += [
        "",
        "## Top global LOBO variants",
        "",
        "| rank | scope | family | pass10 | pass5 | bad | phys fails | min save | avg save | H10 | config |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, v in enumerate(raw["top_global_lobo"][:15], 1):
        s = v["summary"]
        scope = "deploy" if v["config"]["family"] in ("deploy_obs_pose_step_no_risk", "deploy_obs_pose_step_risk") else "diagnostic"
        lines.append(f"| {i} | `{scope}` | `{v['config']['family']}` | `{s['all_pass10']}` | `{s['all_pass5']}` | {s['total_bad']} | {s['phys_fail']} | {pct(s['min_save'])} | {pct(s['avg_save'])} | {s['total_h10']} | `{v['config_id']}` |")
    lines += [
        "",
        "## Interpretation",
        "",
        "This is not a validation or final-test result. It tests whether a verified objective-label mismatch was suppressing safe H10 choices. A passing deployable nested gate would justify a small unused-source overhead-aware confirmation; a diagnostic-only pass would imply missing online risk features; failure would favor richer value/risk training or more targeted boundary acquisition.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    h = raw["headline"]
    block = f"""<!-- {MARKER} -->
## 2026-09-29 vehicle true-variable-H v12 outcome-aligned refit

UTC: {raw['created_utc']}. Development-only IMPROVED relabel/refit over opened banks including v11; no MPC simulation, no validation64, no sealed test, no gradient training. label_flips={raw['label_audit']['flip_count']}; aligned_oracle_save={h['aligned_oracle_save']:.4f}; nested_deploy_save={h['nested_deploy_save']:.4f}, nested_deploy_bad={h['nested_deploy_bad']}, nested_deploy_pass5={h['nested_deploy_pass5']}; nested_all_save={h['nested_all_save']:.4f}, nested_all_bad={h['nested_all_bad']}. Decision: {raw['decision']}. Artifacts: `{rel(OUT / 'summary.md')}`, `{rel(OUT / 'raw.json')}`, `{rel(OUT / 'completed.json')}`.
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
            f.write(f"{STAMP},{NAME},development_outcome_aligned_selector_refit,opened_banks_v0_v1_v2_v8c_v11_no_sim_no_validation_no_test,0,0,{raw['budget_actual']['selector_refit_tree_fits']},0,0,False,{rel(OUT / 'completed.json')},{MARKER}\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    created = now()
    rows_old, hashes, all_groups = load_all_rows()
    rows, label_audit = relabel_rows(rows_old)
    banks = sorted(set(str(r["bank_id"]) for r in rows))
    cfgs_all = cfg_grid()
    cfgs_deploy = [c for c in cfgs_all if c["family"] in ("deploy_obs_pose_step_no_risk", "deploy_obs_pose_step_risk")]
    protocol = {
        "protocol_id": f"{NAME}_preoutcome_frozen_{STAMP}",
        "created_utc": created.isoformat(),
        "classification": "development_IMPROVED_outcome_aligned_label_refit_no_simulation_no_validation_no_test",
        "hypothesis": "v11 exposed old-label objective mismatch: safe/faster H10 was not marked positive when H15 was unsafe. Outcome-aligned relabeling may allow conservative deployable selectors to retain decision-time savings without catastrophic H10 false positives.",
        "before_evidence": [
            "v10b nested deployable risk/value refit over v0/v1/v2/v8c retained 8.43% saving but had 2 catastrophic false positives; global zero-bad configs collapsed below 5% saving.",
            "v11 found 12 source-independent branch states with 8 old positive H10 rows and 2 catastrophic H10 rows; it also found two high-heading-short rows where H15 failed with physical cost above 20k while H10 succeeded near 2k, but the old positive label was false.",
        ],
        "label_definition": {
            "positive": "H10 safe/successful, decision_gain_vs_H15>0, and either H15 unsafe/unsuccessful or H10 physical cost within row tolerance of H15.",
            "catastrophic": "H10 unsafe/unsuccessful, or H15 safe/successful and H10 exceeds H15 physical cost by more than row tolerance.",
            "unchanged_safety_gate": "Nested deployment still requires zero catastrophic H10 false positives and physical gate versus fixed H15 before compute-saving claims.",
        },
        "split": "nested leave-opened-development-bank-out over fresh_v0/fresh_v1/fresh_v2/fresh_v8c/fresh_v11; validation64 and sealed test forbidden",
        "primary_candidate_scope": "pure deployable online features only; scenario/window features are diagnostic and not sufficient for a deployment claim",
        "config_counts": {"all": len(cfgs_all), "deployable": len(cfgs_deploy)},
        "decision_gate": "advance to unused-source overhead-aware confirmation only if nested deployable aggregate has zero catastrophic false positives, physical gate, and >=5% measured branch decision saving; strong if >=10%.",
        "budget_declared": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "gradient_steps": 0, "selector_refit_tree_fits_upper_bound": len(cfgs_all) * len(banks) + len(cfgs_deploy) * len(banks) * (len(banks) - 1) + len(cfgs_all) * len(banks) * (len(banks) - 1) + 2 * len(banks), "validation64_episodes": 0, "sealed_test_episodes": 0},
        "access_flags": {"validation64_bank_opened": False, "sealed_test_accessed": False},
        "hashes": hashes,
    }
    write_json(PROTOCOL, protocol)

    # Label-oracle and fixed-H diagnostics.
    old_rows_for_eval = make_old_label_copy(rows)
    old_oracle = v6.eval_choices(old_rows_for_eval, choices_from_label(rows, "old_h10_beneficial_vs_h15"))
    aligned_oracle = v6.eval_choices(rows, choices_from_label(rows, "h10_beneficial_vs_h15"))
    fixed_h10 = v6.eval_choices(rows, all_h_choices(rows, 10))
    fixed_h15 = v6.eval_choices(rows, all_h_choices(rows, 15))

    global_rows: List[Dict[str, Any]] = []
    fits = 0
    for cfg in cfgs_all:
        hold, nf = eval_lobo(rows, banks, all_groups, cfg)
        fits += nf
        global_rows.append({"config_id": cfg_id(cfg), "config": cfg, "summary": summarize(hold), "holdout_evaluations": hold})
    ranked = sorted(global_rows, key=lambda x: rank_summary(x["summary"]))
    deploy_ranked = sorted([x for x in global_rows if x["config"]["family"] in ("deploy_obs_pose_step_no_risk", "deploy_obs_pose_step_risk")], key=lambda x: rank_summary(x["summary"]))

    nested_deploy, nf = eval_nested(rows, banks, all_groups, cfgs_deploy, "deployable")
    fits += nf
    nested_all, nf = eval_nested(rows, banks, all_groups, cfgs_all, "all_families_diagnostic")
    fits += nf

    nd = nested_deploy["aggregate"]
    na = nested_all["aggregate"]
    headline = {
        "rows": len(rows),
        "banks": banks,
        "old_positive_rows": sum(1 for r in rows if r["old_h10_beneficial_vs_h15"]),
        "aligned_positive_rows": sum(1 for r in rows if r["h10_beneficial_vs_h15"]),
        "old_catastrophic_rows": sum(1 for r in rows if r["old_h10_catastrophic_vs_h15"]),
        "aligned_catastrophic_rows": sum(1 for r in rows if r["h10_catastrophic_vs_h15"]),
        "old_oracle_save": old_oracle["decision_relative_saving_vs_fixed_H15"],
        "old_oracle_physical_gate": bool(old_oracle["physical_gate"]),
        "old_oracle_physical_delta": old_oracle["physical_delta_vs_fixed_H15"],
        "aligned_oracle_save": aligned_oracle["decision_relative_saving_vs_fixed_H15"],
        "aligned_oracle_physical_gate": bool(aligned_oracle["physical_gate"]),
        "aligned_oracle_physical_delta": aligned_oracle["physical_delta_vs_fixed_H15"],
        "fixed_h10_save": fixed_h10["decision_relative_saving_vs_fixed_H15"],
        "fixed_h10_physical_gate": bool(fixed_h10["physical_gate"]),
        "fixed_h10_bad": len(fixed_h10["catastrophic_false_positive_rows"]),
        "global_all_pass10_count": sum(1 for x in global_rows if x["summary"]["all_pass10"]),
        "global_all_pass5_count": sum(1 for x in global_rows if x["summary"]["all_pass5"]),
        "global_deploy_pass10_count": sum(1 for x in deploy_ranked if x["summary"]["all_pass10"]),
        "global_deploy_pass5_count": sum(1 for x in deploy_ranked if x["summary"]["all_pass5"]),
        "best_global_config": ranked[0]["config_id"],
        "best_global_summary": ranked[0]["summary"],
        "best_deploy_config": deploy_ranked[0]["config_id"],
        "best_deploy_summary": deploy_ranked[0]["summary"],
        "nested_deploy_save": nd["decision_relative_saving_vs_fixed_H15"],
        "nested_deploy_bad": len(nd["catastrophic_false_positive_rows"]),
        "nested_deploy_physical_gate": bool(nd["physical_gate"]),
        "nested_deploy_h10": int(nd["chosen_counts"].get("10", 0)),
        "nested_deploy_pass5": bool(nd["pass_5pct_no_cat_fp"]),
        "nested_deploy_pass10": bool(nd["pass_10pct_no_cat_fp"]),
        "nested_all_save": na["decision_relative_saving_vs_fixed_H15"],
        "nested_all_bad": len(na["catastrophic_false_positive_rows"]),
        "nested_all_physical_gate": bool(na["physical_gate"]),
        "nested_all_h10": int(na["chosen_counts"].get("10", 0)),
        "nested_all_pass5": bool(na["pass_5pct_no_cat_fp"]),
        "nested_all_pass10": bool(na["pass_10pct_no_cat_fp"]),
    }
    if headline["nested_deploy_pass10"]:
        decision = "Outcome-aligned deployable refit passes strong opened-bank gate; next freeze a tiny unused-source closed-loop/branch confirmation with actual selector overhead and fixed-H10/H15 comparators before any validation64."
    elif headline["nested_deploy_pass5"]:
        decision = "Outcome-aligned deployable refit passes weak opened-bank gate; next freeze a small unused-source overhead-aware confirmation and require zero catastrophic H10 before scaling."
    elif headline["nested_all_pass5"]:
        decision = "Outcome-aligned labels help only with scenario/window diagnostics, not pure deployable features; next learn/add deployable risk-state features or terminal/risk value estimates before confirmation."
    elif headline["aligned_oracle_save"] > headline["old_oracle_save"] + 0.02:
        decision = "Objective/label mismatch is verified but deployable selector still fails nested gate; next intervention should train/refit richer deployable risk/value representation using the aligned objective, not validation rollout."
    else:
        decision = "Outcome relabeling does not resolve selector failure; next focus on richer risk data/terminal value or reconsider scenario/comparison design."

    raw = {
        "created_utc": now().isoformat(),
        "classification": protocol["classification"],
        "backup_verified_commit_from_supervisor_context": args.backup_verified_commit,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "protocol": rel(PROTOCOL),
        "budget_actual": {"new_mpc_simulation_episodes": 0, "new_control_steps": 0, "new_training_episodes": 0, "gradient_steps": 0, "selector_refit_tree_fits": fits, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "label_audit": label_audit,
        "bank_label_summary": {b: {"rows": sum(1 for r in rows if r["bank_id"] == b), "old_positive": sum(1 for r in rows if r["bank_id"] == b and r["old_h10_beneficial_vs_h15"]), "aligned_positive": sum(1 for r in rows if r["bank_id"] == b and r["h10_beneficial_vs_h15"]), "old_catastrophic": sum(1 for r in rows if r["bank_id"] == b and r["old_h10_catastrophic_vs_h15"]), "aligned_catastrophic": sum(1 for r in rows if r["bank_id"] == b and r["h10_catastrophic_vs_h15"])} for b in banks},
        "headline": headline,
        "old_oracle": old_oracle,
        "aligned_oracle": aligned_oracle,
        "fixed_h10": fixed_h10,
        "fixed_h15": fixed_h15,
        "top_global_lobo": ranked[:25],
        "top_deploy_lobo": deploy_ranked[:25],
        "nested_deploy": nested_deploy,
        "nested_all": nested_all,
        "hashes": {**hashes, rel(PROTOCOL): sha256(PROTOCOL), rel(SOURCE): sha256(SOURCE)},
        "decision": decision,
        "platform": {"python": sys.version, "platform": platform.platform()},
    }
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backup-verified-commit", required=True)
    ap.add_argument("--i-accept-development-outcome-aligned-v12", action="store_true")
    args = ap.parse_args(argv)
    if not args.run or not args.i_accept_development_outcome_aligned_v12:
        raise ContractError("requires --run and explicit v12 development acknowledgement")
    if (OUT / "completed.json").exists():
        done = read_json(OUT / "completed.json")
        print(json.dumps({"already_completed": rel(OUT / "completed.json"), "headline": done.get("headline"), "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    raw = run(args)
    write_json(OUT / "raw.json", raw)
    write_summary(raw)
    req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_OUTCOME_ALIGNED_REFIT_V12_{STAMP}.json"
    write_json(req, {"created_utc": raw["created_utc"], "reason": "backup after v12 outcome-aligned refit before any further science", "paths": [rel(OUT), rel(PROTOCOL), rel(STATE), rel(SOURCE), "STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md", "EXPERIMENT_REGISTRY.csv"], "validation64_bank_opened": False, "sealed_test_accessed": False})
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
        "new_training_episodes": 0,
        "gradient_steps": 0,
        "selector_refit_tree_fits": raw["budget_actual"]["selector_refit_tree_fits"],
        "hashes": {rel(SOURCE): sha256(SOURCE), rel(PROTOCOL): sha256(PROTOCOL), rel(OUT / "raw.json"): sha256(OUT / "raw.json"), rel(OUT / "summary.md"): sha256(OUT / "summary.md"), rel(req): sha256(req)},
    }
    write_json(OUT / "completed.json", done)
    append_docs(raw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text("# Continue state after v12 outcome-aligned refit\n\n" + json.dumps(clean({"utc": raw["created_utc"], "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "artifacts": {"summary": rel(OUT / "summary.md"), "raw": rel(OUT / "raw.json"), "completed": rel(OUT / "completed.json"), "protocol": rel(PROTOCOL), "backup_request": rel(req)}, "current_backup_status": "not verified after v12; run backup before further science", "next_action": "After backup, follow v12 decision. If nested deployable passed, freeze unused-source overhead-aware confirmation; otherwise train/refit richer deployable risk/value representation under aligned objective. No validation64/sealed test."}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": rel(OUT / "completed.json"), "summary": rel(OUT / "summary.md"), "headline": raw["headline"], "decision": raw["decision"], "budget_actual": raw["budget_actual"], "validation64_bank_opened": False, "sealed_test_accessed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "failed.json", {"passed": False, "status": "failed", "error": type(exc).__name__, "message": str(exc), "validation64_bank_opened": False, "sealed_test_accessed": False})
        raise
