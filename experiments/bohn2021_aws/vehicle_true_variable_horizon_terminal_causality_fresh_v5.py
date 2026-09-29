#!/usr/bin/env python3
"""Fresh-bank terminal-causality diagnostic for true-variable-H H10/H15.

Development-only, no-simulation diagnostic.  It uses only already-opened
fresh-source banks v0/v1/v2 to answer a causal design question raised by the
recent selector/value-refit failures: are the catastrophic H10 rows primarily a
terminal-profile/objective mismatch, or are they intrinsic short-horizon risk
modes that require better online state representation/training?

No validation64 or sealed final-test files are read.  No MPC rollout, gradient
training, checkpoint writing, or selector refit is performed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
NAME = "vehicle_true_variable_horizon_terminal_causality_fresh_v5"
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / f"research_artifacts/aws_diagnostics/{NAME}_{STAMP}"
STATE = ROOT / f"research_artifacts/aws_state/continue_state_{STAMP}_after_terminal_causality_fresh_v5.md"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = f"vehicle-true-variable-H-terminal-causality-fresh-v5-{STAMP}"
FIRST_SUPERVISOR_EVENT = dt.datetime.fromisoformat("2026-09-26T10:55:29.419331+00:00")
BANK_DIRS = {
    "fresh_v0": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z",
    "fresh_v1": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z",
    "fresh_v2": ROOT / "research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z",
}
PROFILES = ["shared_h15_terminal", "matched_terminal"]
MIN_ROW_TOL = 2.0
MATERIAL_TERMINAL_EFFECT = 3.0


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


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


def med_h(row: Mapping[str, Any], h: int) -> Mapping[str, Any]:
    med = row.get("median_by_h") or {}
    return med.get(str(h)) or med.get(h) or {}


def stats(vals: Iterable[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    mid = xs[len(xs)//2] if len(xs) % 2 else 0.5 * (xs[len(xs)//2 - 1] + xs[len(xs)//2])
    return {"n": len(xs), "min": xs[0], "median": mid, "mean": sum(xs) / len(xs), "max": xs[-1]}


def assert_dev_only(raw: Mapping[str, Any], done: Mapping[str, Any], bank: str) -> None:
    for name, obj in (("raw", raw), ("completed", done)):
        for flag in ("sealed_test_accessed", "sealed_test_bank_opened", "test_accessed", "validation64_bank_opened"):
            if obj.get(flag) is True:
                raise ContractError(f"forbidden {flag}=True in {bank} {name}")
    if done.get("passed") is not True and done.get("hard_pass") is not True and done.get("status") not in ("complete", "completed"):
        raise ContractError(f"bank did not complete cleanly: {bank}")


def profile_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    h10 = med_h(row, 10); h15 = med_h(row, 15)
    tol = max(MIN_ROW_TOL, sf(row.get("row_physical_tolerance_vs_H15", row.get("row_physical_tolerance")), MIN_ROW_TOL))
    h10_phys = sf(h10.get("physical")); h15_phys = sf(h15.get("physical"))
    h10_dec = sf(h10.get("decision_sum_s")); h15_dec = sf(h15.get("decision_sum_s"))
    h10_sol = sf(h10.get("solver_sum_s")); h15_sol = sf(h15.get("solver_sum_s"))
    h10_safe = bool(h10.get("safe_all")); h15_safe = bool(h15.get("safe_all"))
    phys_delta = h10_phys - h15_phys
    decision_gain = h15_dec - h10_dec
    solver_gain = h15_sol - h10_sol
    beneficial = bool(h10_safe and h15_safe and phys_delta <= tol and decision_gain > 0.0)
    catastrophic = bool((not h10_safe) or (h15_safe and phys_delta > tol))
    return {
        "terminal_profile": str(row.get("terminal_profile")),
        "h10_physical": h10_phys,
        "h15_physical": h15_phys,
        "h10_decision_sum_s": h10_dec,
        "h15_decision_sum_s": h15_dec,
        "h10_solver_sum_s": h10_sol,
        "h15_solver_sum_s": h15_sol,
        "h10_safe_all": h10_safe,
        "h15_safe_all": h15_safe,
        "phys_delta_h10_minus_h15": phys_delta,
        "decision_gain_h10_vs_h15_s": decision_gain,
        "solver_gain_h10_vs_h15_s": solver_gain,
        "row_physical_tolerance": tol,
        "h10_beneficial_vs_h15": beneficial,
        "h10_catastrophic_vs_h15": catastrophic,
    }


def load_profile_pairs() -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    hashes: Dict[str, str] = {rel(Path(__file__)): sha256(Path(__file__))}
    pairs: List[Dict[str, Any]] = []
    for bank, d in BANK_DIRS.items():
        raw_path = d / "raw.json"; done_path = d / "completed.json"
        raw = read_json(raw_path); done = read_json(done_path)
        assert_dev_only(raw, done, bank)
        hashes[rel(raw_path)] = sha256(raw_path); hashes[rel(done_path)] = sha256(done_path)
        rows = ((raw.get("analysis") or {}).get("state_profile_rows") or [])
        by_state: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
        meta: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            profile = str(row.get("terminal_profile"))
            if profile not in PROFILES:
                continue
            base = str(row.get("base_state_id") or row.get("state_id"))
            by_state[base][profile] = row
            slot = si(row.get("branch_state_slot"), 0)
            meta.setdefault(base, {
                "bank_id": bank,
                "base_state_id": base,
                "fresh_case_index": row.get("fresh_case_index"),
                "source_candidate_index": row.get("source_candidate_index"),
                "group": str(row.get("fresh_confirmation_group") or "unknown"),
                "window": "early_mid" if slot == 0 else "mid_late",
                "branch_state_slot": slot,
            })
        for base, profs in by_state.items():
            if not all(p in profs for p in PROFILES):
                raise ContractError(f"missing paired terminal profile for {bank} {base}: {sorted(profs)}")
            shared = profile_row(profs["shared_h15_terminal"])
            matched = profile_row(profs["matched_terminal"])
            item = dict(meta[base])
            item.update({
                "shared_h15_terminal": shared,
                "matched_terminal": matched,
                "shared_label": 10 if shared["h10_beneficial_vs_h15"] else 15,
                "matched_label": 10 if matched["h10_beneficial_vs_h15"] else 15,
                "both_beneficial": bool(shared["h10_beneficial_vs_h15"] and matched["h10_beneficial_vs_h15"]),
                "either_catastrophic": bool(shared["h10_catastrophic_vs_h15"] or matched["h10_catastrophic_vs_h15"]),
                "terminal_label_flip": bool(shared["h10_beneficial_vs_h15"] != matched["h10_beneficial_vs_h15"]),
                "h10_shared_minus_matched_physical": shared["h10_physical"] - matched["h10_physical"],
                "h15_shared_minus_matched_physical": shared["h15_physical"] - matched["h15_physical"],
                "h10_shared_minus_matched_decision_s": shared["h10_decision_sum_s"] - matched["h10_decision_sum_s"],
                "h15_shared_minus_matched_decision_s": shared["h15_decision_sum_s"] - matched["h15_decision_sum_s"],
            })
            pairs.append(item)
    if len(pairs) != 48:
        raise ContractError(f"expected 48 fresh terminal-profile pairs, got {len(pairs)}")
    return pairs, hashes


def eval_policy(pairs: Sequence[Mapping[str, Any]], profile: str, choices: Mapping[Tuple[str, str], int]) -> Dict[str, Any]:
    fixed_phys = fixed_dec = fixed_sol = tol_sum = 0.0
    pol_phys = pol_dec = pol_sol = 0.0
    counts = Counter(); conf = Counter(); bad = []; unsafe = []
    for r in pairs:
        pr = r[profile]
        fixed_phys += sf(pr["h15_physical"]); fixed_dec += sf(pr["h15_decision_sum_s"]); fixed_sol += sf(pr["h15_solver_sum_s"]); tol_sum += sf(pr["row_physical_tolerance"])
        h = int(choices.get((str(r["bank_id"]), str(r["base_state_id"])), 15))
        counts[str(h)] += 1
        positive = bool(pr["h10_beneficial_vs_h15"])
        if h == 10:
            pol_phys += sf(pr["h10_physical"]); pol_dec += sf(pr["h10_decision_sum_s"]); pol_sol += sf(pr["h10_solver_sum_s"])
            conf["TP" if positive else "FP"] += 1
            if not pr["h10_safe_all"]:
                unsafe.append({"bank_id": r["bank_id"], "base_state_id": r["base_state_id"], "group": r["group"], "window": r["window"]})
            if pr["h10_catastrophic_vs_h15"]:
                bad.append({"bank_id": r["bank_id"], "base_state_id": r["base_state_id"], "group": r["group"], "window": r["window"], "phys_delta": sf(pr["phys_delta_h10_minus_h15"]), "tol": sf(pr["row_physical_tolerance"])})
        else:
            pol_phys += sf(pr["h15_physical"]); pol_dec += sf(pr["h15_decision_sum_s"]); pol_sol += sf(pr["h15_solver_sum_s"])
            conf["FN" if positive else "TN"] += 1
    dec_save = (fixed_dec - pol_dec) / fixed_dec if fixed_dec > 0 else 0.0
    sol_save = (fixed_sol - pol_sol) / fixed_sol if fixed_sol > 0 else 0.0
    phys_delta = pol_phys - fixed_phys
    return {
        "profile": profile,
        "groups": len(pairs),
        "chosen_counts": dict(counts),
        "confusion": dict(conf),
        "physical_delta_vs_fixed_H15": phys_delta,
        "physical_tolerance_sum": tol_sum,
        "decision_relative_saving_vs_fixed_H15": dec_save,
        "solver_relative_saving_vs_fixed_H15": sol_save,
        "physical_gate": phys_delta <= tol_sum,
        "catastrophic_false_positive_rows": bad,
        "unsafe_rows": unsafe,
        "pass_5pct_no_cat_fp": bool(dec_save >= 0.05 and phys_delta <= tol_sum and not bad and not unsafe),
        "pass_10pct_no_cat_fp": bool(dec_save >= 0.10 and phys_delta <= tol_sum and not bad and not unsafe),
    }


def pct(x: Any) -> str:
    return "None" if x is None else f"{100.0 * sf(x):.2f}%"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, hashes = load_profile_pairs()

    headline = {
        "paired_states": len(pairs),
        "shared_beneficial": sum(1 for r in pairs if r["shared_h15_terminal"]["h10_beneficial_vs_h15"]),
        "matched_beneficial": sum(1 for r in pairs if r["matched_terminal"]["h10_beneficial_vs_h15"]),
        "both_beneficial": sum(1 for r in pairs if r["both_beneficial"]),
        "terminal_label_flips": sum(1 for r in pairs if r["terminal_label_flip"]),
        "shared_catastrophic": sum(1 for r in pairs if r["shared_h15_terminal"]["h10_catastrophic_vs_h15"]),
        "matched_catastrophic": sum(1 for r in pairs if r["matched_terminal"]["h10_catastrophic_vs_h15"]),
        "shared_cat_resolved_by_matched_noncat": sum(1 for r in pairs if r["shared_h15_terminal"]["h10_catastrophic_vs_h15"] and not r["matched_terminal"]["h10_catastrophic_vs_h15"]),
        "shared_cat_persists_under_matched": sum(1 for r in pairs if r["shared_h15_terminal"]["h10_catastrophic_vs_h15"] and r["matched_terminal"]["h10_catastrophic_vs_h15"]),
        "matched_cat_new_vs_shared": sum(1 for r in pairs if r["matched_terminal"]["h10_catastrophic_vs_h15"] and not r["shared_h15_terminal"]["h10_catastrophic_vs_h15"]),
        "material_h10_terminal_effect_rows": sum(1 for r in pairs if abs(sf(r["h10_shared_minus_matched_physical"])) >= MATERIAL_TERMINAL_EFFECT),
    }

    choices_oracle_shared = {(r["bank_id"], r["base_state_id"]): (10 if r["shared_h15_terminal"]["h10_beneficial_vs_h15"] else 15) for r in pairs}
    choices_oracle_matched = {(r["bank_id"], r["base_state_id"]): (10 if r["matched_terminal"]["h10_beneficial_vs_h15"] else 15) for r in pairs}
    choices_both = {(r["bank_id"], r["base_state_id"]): (10 if r["both_beneficial"] else 15) for r in pairs}
    choices_all_h10 = {(r["bank_id"], r["base_state_id"]): 10 for r in pairs}
    policy_evals = {
        "shared_row_oracle_on_shared": eval_policy(pairs, "shared_h15_terminal", choices_oracle_shared),
        "matched_row_oracle_on_matched": eval_policy(pairs, "matched_terminal", choices_oracle_matched),
        "matched_labels_transferred_to_shared": eval_policy(pairs, "shared_h15_terminal", choices_oracle_matched),
        "shared_labels_transferred_to_matched": eval_policy(pairs, "matched_terminal", choices_oracle_shared),
        "terminal_agreement_both_beneficial_on_shared": eval_policy(pairs, "shared_h15_terminal", choices_both),
        "terminal_agreement_both_beneficial_on_matched": eval_policy(pairs, "matched_terminal", choices_both),
        "fixed_H10_on_shared": eval_policy(pairs, "shared_h15_terminal", choices_all_h10),
        "fixed_H10_on_matched": eval_policy(pairs, "matched_terminal", choices_all_h10),
    }

    by_group: Dict[str, Dict[str, Any]] = {}
    tmp: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in pairs:
        tmp[f"{r['group']}::{r['window']}"].append(r)
    for key, rs in sorted(tmp.items()):
        by_group[key] = {
            "n": len(rs),
            "shared_beneficial": sum(1 for r in rs if r["shared_h15_terminal"]["h10_beneficial_vs_h15"]),
            "matched_beneficial": sum(1 for r in rs if r["matched_terminal"]["h10_beneficial_vs_h15"]),
            "both_beneficial": sum(1 for r in rs if r["both_beneficial"]),
            "shared_cat": sum(1 for r in rs if r["shared_h15_terminal"]["h10_catastrophic_vs_h15"]),
            "matched_cat": sum(1 for r in rs if r["matched_terminal"]["h10_catastrophic_vs_h15"]),
            "label_flips": sum(1 for r in rs if r["terminal_label_flip"]),
            "h10_shared_minus_matched_physical": stats([sf(r["h10_shared_minus_matched_physical"]) for r in rs]),
        }

    terminal_effect_rows = sorted([
        {
            "bank_id": r["bank_id"], "base_state_id": r["base_state_id"], "group": r["group"], "window": r["window"],
            "shared_label": r["shared_label"], "matched_label": r["matched_label"],
            "shared_cat": r["shared_h15_terminal"]["h10_catastrophic_vs_h15"],
            "matched_cat": r["matched_terminal"]["h10_catastrophic_vs_h15"],
            "h10_shared_minus_matched_physical": sf(r["h10_shared_minus_matched_physical"]),
            "shared_phys_delta": sf(r["shared_h15_terminal"]["phys_delta_h10_minus_h15"]),
            "matched_phys_delta": sf(r["matched_terminal"]["phys_delta_h10_minus_h15"]),
            "shared_decision_gain_s": sf(r["shared_h15_terminal"]["decision_gain_h10_vs_h15_s"]),
            "matched_decision_gain_s": sf(r["matched_terminal"]["decision_gain_h10_vs_h15_s"]),
        }
        for r in pairs if abs(sf(r["h10_shared_minus_matched_physical"])) >= MATERIAL_TERMINAL_EFFECT or r["terminal_label_flip"] or r["either_catastrophic"]
    ], key=lambda x: (-abs(sf(x["h10_shared_minus_matched_physical"])), x["bank_id"], x["base_state_id"]))

    if headline["shared_catastrophic"] and headline["shared_cat_resolved_by_matched_noncat"] >= 0.5 * headline["shared_catastrophic"]:
        next_decision = "Terminal-profile/objective mismatch is a primary causal suspect: most shared-H15 H10 catastrophics become non-catastrophic under matched-H terminal rollouts. Next run a bounded terminal-value/objective refit or terminal-feature representation smoke, not another nearest-neighbour selector sweep."
    elif headline["shared_cat_persists_under_matched"] >= headline["shared_cat_resolved_by_matched_noncat"]:
        next_decision = "Intrinsic short-H risk dominates the fresh-bank catastrophics. Next intervention should target state representation/safety-value prediction or exploration of risk states before terminal refit."
    else:
        next_decision = "Terminal and intrinsic risk are mixed; design a bounded two-arm intervention separating terminal-value calibration from online risk representation."

    raw = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "development_IMPROVED_fresh_bank_terminal_causality_no_simulation_no_validation_no_test",
        "hypothesis": "If catastrophic H10 rows disappear under matched-H terminal profiles for the same states, terminal/objective mismatch is limiting; if they persist, online risk representation/training is the limiting factor.",
        "budgets": {"new_simulations": 0, "new_training_episodes": 0, "gradient_steps": 0, "selector_refits": 0, "validation64_episodes": 0, "sealed_test_episodes": 0},
        "headline": headline,
        "policy_evaluations": policy_evals,
        "by_group_window": by_group,
        "terminal_effect_or_flip_rows": terminal_effect_rows,
        "hashes": hashes,
        "access_flags": {"sealed_test_accessed": False, "validation64_bank_opened": False, "mobile_robot_mppi_resumed": False},
        "next_decision": next_decision,
        "platform": {"python": sys.version, "platform": platform.platform()},
    }

    raw_path = OUT / "raw.json"; summary_path = OUT / "summary.md"; done_path = OUT / "completed.json"
    write_json(raw_path, raw)

    lines: List[str] = []
    lines += [
        "# Vehicle true-variable-H terminal-causality fresh-bank v5",
        "",
        f"UTC: `{raw['created_utc']}`. Development-only no-simulation analysis of already-opened fresh_v0/v1/v2 true-H10/H15 banks; validation64 and sealed test remained closed.",
        "",
        "## Headline",
        "",
        f"- Paired states: `{headline['paired_states']}`; shared-H15 beneficial `{headline['shared_beneficial']}`, matched-terminal beneficial `{headline['matched_beneficial']}`, both-beneficial `{headline['both_beneficial']}`.",
        f"- Terminal label flips: `{headline['terminal_label_flips']}`; material H10 terminal-effect rows (|shared-matched physical| >= {MATERIAL_TERMINAL_EFFECT}): `{headline['material_h10_terminal_effect_rows']}`.",
        f"- Shared-H15 catastrophic rows: `{headline['shared_catastrophic']}`; resolved by matched-terminal non-cat `{headline['shared_cat_resolved_by_matched_noncat']}`; persistent under matched `{headline['shared_cat_persists_under_matched']}`; matched-only catastrophics `{headline['matched_cat_new_vs_shared']}`.",
        "",
        "## Policy/oracle implications across all fresh banks",
        "",
        "| policy | profile evaluated | H counts | bad rows | phys Δ/tol | decision save | solver save | pass10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, ev in policy_evals.items():
        lines.append(f"| `{name}` | `{ev['profile']}` | `{ev['chosen_counts']}` | {len(ev['catastrophic_false_positive_rows'])} | {ev['physical_delta_vs_fixed_H15']:.4g}/{ev['physical_tolerance_sum']:.4g} | {pct(ev['decision_relative_saving_vs_fixed_H15'])} | {pct(ev['solver_relative_saving_vs_fixed_H15'])} | `{ev['pass_10pct_no_cat_fp']}` |")
    lines += ["", "## Group/window anatomy", "", "| group/window | n | shared ben | matched ben | both ben | shared cat | matched cat | flips | median H10 shared-matched phys |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, g in by_group.items():
        lines.append(f"| `{key}` | {g['n']} | {g['shared_beneficial']} | {g['matched_beneficial']} | {g['both_beneficial']} | {g['shared_cat']} | {g['matched_cat']} | {g['label_flips']} | {g['h10_shared_minus_matched_physical']['median'] if g['h10_shared_minus_matched_physical']['median'] is not None else 'NA'} |")
    lines += ["", "## Material terminal-effect / flip rows (top)", "", "| bank | state | group/window | shared label | matched label | shared cat | matched cat | H10 shared-matched phys | shared Δ | matched Δ |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in terminal_effect_rows[:24]:
        lines.append(f"| `{r['bank_id']}` | `{r['base_state_id']}` | `{r['group']}::{r['window']}` | {r['shared_label']} | {r['matched_label']} | `{r['shared_cat']}` | `{r['matched_cat']}` | {r['h10_shared_minus_matched_physical']:.4g} | {r['shared_phys_delta']:.4g} | {r['matched_phys_delta']:.4g} |")
    lines += ["", "## Decision", "", next_decision, "", "This diagnostic is not validation and not ORIGINAL SAC reproduction. It diagnoses whether the next bounded intervention should be terminal/objective/value refit versus online risk-representation/training. It uses measured decision/solver timings already produced by the fresh banks and does not infer speed from nominal H alone."]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    backup_req = BACKUP_DIR / f"REQUEST_BACKUP_AFTER_TERMINAL_CAUSALITY_FRESH_V5_{STAMP}.json"
    write_json(backup_req, {"created_utc": raw["created_utc"], "reason": "backup after fresh-bank terminal-causality diagnostic v5", "paths": [rel(raw_path), rel(summary_path), rel(done_path)], "sealed_test_accessed": False, "validation64_bank_opened": False})
    done = {"passed": True, "status": "complete", "marker": MARKER, "summary": rel(summary_path), "raw": rel(raw_path), "backup_request": rel(backup_req), "headline": headline, "sealed_test_accessed": False, "validation64_bank_opened": False, "new_simulations": 0, "gradient_steps": 0, "selector_refits": 0, "hashes": {rel(Path(__file__)): sha256(Path(__file__)), rel(raw_path): sha256(raw_path), rel(summary_path): sha256(summary_path)}}
    write_json(done_path, done)

    state_text = f"""# Continue state after {NAME}

UTC: {raw['created_utc']}

Result: shared_cat={headline['shared_catastrophic']}, resolved_by_matched={headline['shared_cat_resolved_by_matched_noncat']}, persistent={headline['shared_cat_persists_under_matched']}, label_flips={headline['terminal_label_flips']}, both_beneficial={headline['both_beneficial']}.

Access/budget: no simulations, no validation64, no sealed test, no gradient training, no selector refit.

Next decision: {next_decision}

Artifacts: {rel(summary_path)}, {rel(raw_path)}, {rel(done_path)}.
"""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(state_text, encoding="utf-8")
    audit_line = f"\n- {raw['created_utc']} `{NAME}`: fresh-bank terminal-causality diagnostic; shared_cat={headline['shared_catastrophic']}, resolved_by_matched={headline['shared_cat_resolved_by_matched_noncat']}, persistent={headline['shared_cat_persists_under_matched']}, flips={headline['terminal_label_flips']}; no simulation/validation/test; artifacts `{rel(summary_path)}`.\n"
    for doc in ["STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"]:
        p = ROOT / doc
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if MARKER not in old:
            p.write_text(old.rstrip() + audit_line, encoding="utf-8")
    with (ROOT / "EXPERIMENT_REGISTRY.csv").open("a", encoding="utf-8") as f:
        f.write(f"{STAMP},{NAME},development_terminal_causality_fresh_banks,no_simulation_no_rng,opened_fresh_v0_v1_v2_no_validation_no_test,0,0,0,0,0,False,{rel(done_path)},{MARKER}\n")

    print("SUMMARY " + rel(summary_path))
    print(json.dumps({"headline": headline, "next_decision": next_decision, "policy_key_results": {k: {"decision_save": v["decision_relative_saving_vs_fixed_H15"], "bad": len(v["catastrophic_false_positive_rows"]), "pass10": v["pass_10pct_no_cat_fp"], "h_counts": v["chosen_counts"]} for k, v in policy_evals.items() if k in ("shared_row_oracle_on_shared", "matched_row_oracle_on_matched", "terminal_agreement_both_beneficial_on_shared", "matched_labels_transferred_to_shared")}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "completed.json", {"passed": False, "status": "failed", "marker": MARKER, "error": type(exc).__name__, "message": str(exc), "sealed_test_accessed": False, "validation64_bank_opened": False})
        raise
