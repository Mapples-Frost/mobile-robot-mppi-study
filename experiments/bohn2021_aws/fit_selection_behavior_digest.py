#!/usr/bin/env python3
"""Compact diagnostic of latency-tree fit-vs-selection behavior.

Reads only previously generated training-metadata diagnostics and training-run
registration files. It performs no simulations and deliberately does not touch
validation or sealed test artifacts.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

RAW_DIAG = Path("research_artifacts/aws_diagnostics/fit_population_diagnosis/raw.json")
TRAIN_BASE = Path("research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/train")
OUT_DIR = Path("research_artifacts/aws_diagnostics/fit_selection_behavior_digest")
OUT_JSON = OUT_DIR / "raw.json"
OUT_MD = OUT_DIR / "summary.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_rank_value(cand: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not cand:
        return None
    rank = cand.get("rank") or {}
    val = rank.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def objective(cand: Dict[str, Any]) -> float:
    val = safe_rank_value(cand, "objective")
    if val is None:
        return float("inf")
    return val


def compact_candidate(cand: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if cand is None:
        return None
    return {
        "candidate": cand.get("candidate"),
        "eligible": cand.get("eligible"),
        "constant_H25_behavior": cand.get("constant_H25_behavior"),
        "within_episode_switches": cand.get("within_episode_switches"),
        "nominal_leaf_actions": cand.get("nominal_leaf_actions"),
        "episodes": cand.get("episodes"),
        "steps": cand.get("steps"),
        "success": cand.get("success"),
        "constraints": cand.get("constraints"),
        "solver_failures": cand.get("solver_failures"),
        "objective": safe_rank_value(cand, "objective"),
        "cost_change": safe_rank_value(cand, "cost_change"),
        "physical_change": safe_rank_value(cand, "physical_change"),
        "time_ratio": safe_rank_value(cand, "time_ratio"),
        "violations": (cand.get("rank") or {}).get("violations"),
    }


def min_candidate(items: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    items = list(items)
    if not items:
        return None
    return min(items, key=objective)


def summarize_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"type": type(policy).__name__}
    out: Dict[str, Any] = {"keys": sorted(policy.keys())}
    for key in ("type", "policy", "candidate", "selected", "selected_policy", "H", "horizon", "tree", "rules"):
        if key in policy:
            val = policy[key]
            if isinstance(val, (str, int, float, bool)) or val is None:
                out[key] = val
            elif isinstance(val, list):
                out[key] = {"type": "list", "len": len(val), "preview": val[:3]}
            elif isinstance(val, dict):
                out[key] = {"type": "dict", "keys": sorted(val.keys())[:20], "len": len(val)}
    return out


def selection_registration_digest(reg: Any) -> Dict[str, Any]:
    """Return schema/decision hints without dumping whole registration file."""
    digest: Dict[str, Any] = {"type": type(reg).__name__}
    if not isinstance(reg, dict):
        return digest
    digest["top_level_keys"] = sorted(reg.keys())
    for key in (
        "selected_policy",
        "selected",
        "selected_candidate",
        "fallback",
        "fallback_reason",
        "finalist_ids",
        "finalists",
        "baseline",
        "fixed",
        "metric",
        "objective",
        "selection_metric",
    ):
        if key in reg:
            val = reg[key]
            if isinstance(val, (str, int, float, bool)) or val is None:
                digest[key] = val
            elif isinstance(val, list):
                digest[key] = {"type": "list", "len": len(val), "preview": val[:5]}
            elif isinstance(val, dict):
                digest[key] = {"type": "dict", "keys": sorted(val.keys())[:30], "len": len(val)}
    # Search shallowly for scalar fields whose path names indicate selection decisions.
    hits: List[Dict[str, Any]] = []
    terms = ("selected", "fallback", "finalist", "objective", "score", "fixed")

    def walk(obj: Any, path: str, depth: int) -> None:
        if len(hits) >= 80 or depth < 0:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else str(k)
                if any(t in str(k).lower() for t in terms):
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        hits.append({"path": p, "value": v})
                    elif isinstance(v, list):
                        preview = v[:4]
                        hits.append({"path": p, "type": "list", "len": len(v), "preview": preview})
                    elif isinstance(v, dict):
                        hits.append({"path": p, "type": "dict", "keys": sorted(v.keys())[:12], "len": len(v)})
                walk(v, p, depth - 1)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:12]):
                walk(v, f"{path}[{i}]", depth - 1)

    walk(reg, "", 4)
    digest["decision_field_hits"] = hits
    return digest


def main() -> int:
    if not RAW_DIAG.exists():
        raise FileNotFoundError(f"missing input diagnostic: {RAW_DIAG}")
    raw = load_json(RAW_DIAG)
    seed_blocks = raw.get("all_seeds") if isinstance(raw, dict) else None
    if not isinstance(seed_blocks, list):
        raise ValueError("fit_population_diagnosis/raw.json lacks all_seeds list")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    digest: Dict[str, Any] = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "method": "IMPROVED latency-tree training metadata digest; no simulation",
        "inputs": {
            "fit_population_diagnosis_raw": str(RAW_DIAG),
            "fit_population_diagnosis_raw_sha256": sha256_file(RAW_DIAG),
            "train_base": str(TRAIN_BASE),
        },
        "test_accessed": False,
        "validation_accessed": False,
        "new_simulations": 0,
        "seeds": [],
        "cross_seed_findings": {},
    }

    selected_constant_or_fixed = 0
    selected_switching = 0
    total_eligible = 0
    total_eligible_constant = 0
    total_eligible_switching = 0
    constant_negative_objective = 0
    switching_negative_objective = 0
    seeds_with_switching_opportunity = 0
    seeds_where_selection_not_best_training_objective = 0
    seeds_where_best_switching_beats_selected_candidate = 0

    for block in seed_blocks:
        seed = int(block.get("seed"))
        candidates = block.get("candidates") or []
        by_id = {c.get("candidate"): c for c in candidates if isinstance(c, dict)}
        eligible = [c for c in candidates if c.get("eligible")]
        eligible_const = [c for c in eligible if c.get("constant_H25_behavior")]
        eligible_switch = [c for c in eligible if not c.get("constant_H25_behavior") and (c.get("within_episode_switches") or 0) > 0]
        finalists = [by_id.get(cid) for cid in (block.get("finalist_ids") or [])]
        finalists_compact = [compact_candidate(c) for c in finalists]
        selected_id = block.get("selected_policy")
        selected_candidate = by_id.get(selected_id)
        selected_is_fixed = selected_id == "fixed" or selected_candidate is None
        selected_is_constant = bool(selected_is_fixed or (selected_candidate and selected_candidate.get("constant_H25_behavior")))
        selected_is_switching = bool(selected_candidate and (not selected_candidate.get("constant_H25_behavior")) and (selected_candidate.get("within_episode_switches") or 0) > 0)

        best_eligible = min_candidate(eligible)
        best_const = min_candidate(eligible_const)
        best_switch = min_candidate(eligible_switch)
        best_finalist = min_candidate([c for c in finalists if c])

        total_eligible += len(eligible)
        total_eligible_constant += len(eligible_const)
        total_eligible_switching += len(eligible_switch)
        constant_negative_objective += sum(1 for c in eligible_const if objective(c) < 0)
        switching_negative_objective += sum(1 for c in eligible_switch if objective(c) < 0)
        if eligible_switch:
            seeds_with_switching_opportunity += 1
        if selected_is_constant:
            selected_constant_or_fixed += 1
        if selected_is_switching:
            selected_switching += 1
        if selected_candidate and best_eligible and selected_candidate.get("candidate") != best_eligible.get("candidate"):
            seeds_where_selection_not_best_training_objective += 1
        if selected_candidate and best_switch and objective(best_switch) < objective(selected_candidate):
            seeds_where_best_switching_beats_selected_candidate += 1

        seed_dir = TRAIN_BASE / f"vehicle_s{seed}"
        file_hashes: Dict[str, Optional[str]] = {}
        for fname in ("fit.json", "completed.json", "selection_registration.json", "policy.json"):
            p = seed_dir / fname
            file_hashes[fname] = sha256_file(p) if p.exists() else None
        policy_summary = summarize_policy(load_json(seed_dir / "policy.json")) if (seed_dir / "policy.json").exists() else {"missing": True}
        selection_summary = selection_registration_digest(load_json(seed_dir / "selection_registration.json")) if (seed_dir / "selection_registration.json").exists() else {"missing": True}

        seed_digest = {
            "seed": seed,
            "file_hashes": file_hashes,
            "candidate_counts": {
                "fit_candidates": len(candidates),
                "eligible": len(eligible),
                "eligible_constant_H25": len(eligible_const),
                "eligible_with_switches": len(eligible_switch),
                "eligible_constant_fraction": (len(eligible_const) / len(eligible)) if eligible else None,
                "eligible_switching_fraction": (len(eligible_switch) / len(eligible)) if eligible else None,
                "eligible_constant_negative_objective": sum(1 for c in eligible_const if objective(c) < 0),
                "eligible_switching_negative_objective": sum(1 for c in eligible_switch if objective(c) < 0),
            },
            "selected_policy": selected_id,
            "selected_is_fixed_or_missing_candidate": selected_is_fixed,
            "selected_constant_or_fixed": selected_is_constant,
            "selected_switching": selected_is_switching,
            "selected_candidate": compact_candidate(selected_candidate),
            "finalist_ids": block.get("finalist_ids"),
            "finalists": finalists_compact,
            "finalist_counts": {
                "constant_H25": sum(1 for c in finalists if c and c.get("constant_H25_behavior")),
                "switching": sum(1 for c in finalists if c and (not c.get("constant_H25_behavior")) and (c.get("within_episode_switches") or 0) > 0),
                "missing": sum(1 for c in finalists if c is None),
            },
            "best_eligible_by_training_objective": compact_candidate(best_eligible),
            "best_constant_by_training_objective": compact_candidate(best_const),
            "best_switching_by_training_objective": compact_candidate(best_switch),
            "best_finalist_by_training_objective": compact_candidate(best_finalist),
            "selection_vs_training_objective_flags": {
                "selected_is_best_eligible_training_objective": bool(selected_candidate and best_eligible and selected_candidate.get("candidate") == best_eligible.get("candidate")),
                "selected_is_best_finalist_training_objective": bool(selected_candidate and best_finalist and selected_candidate.get("candidate") == best_finalist.get("candidate")),
                "best_switching_training_objective_beats_selected_candidate": bool(selected_candidate and best_switch and objective(best_switch) < objective(selected_candidate)),
                "best_switching_training_objective_beats_best_constant": bool(best_switch and best_const and objective(best_switch) < objective(best_const)),
                "fixed_selected_despite_candidate_finalists": bool(selected_is_fixed and finalists),
            },
            "policy_json_summary": policy_summary,
            "selection_registration_digest": selection_summary,
        }
        digest["seeds"].append(seed_digest)

        print(json.dumps({
            "seed": seed,
            "selected_policy": selected_id,
            "selected_constant_or_fixed": selected_is_constant,
            "selected_switching": selected_is_switching,
            "eligible_constant_H25": len(eligible_const),
            "eligible_with_switches": len(eligible_switch),
            "best_eligible": compact_candidate(best_eligible),
            "best_switching": compact_candidate(best_switch),
            "best_finalist": compact_candidate(best_finalist),
            "flags": seed_digest["selection_vs_training_objective_flags"],
        }, sort_keys=True))

    nseeds = len(seed_blocks)
    digest["cross_seed_findings"] = {
        "seeds": nseeds,
        "total_eligible": total_eligible,
        "total_eligible_constant_H25": total_eligible_constant,
        "total_eligible_with_switches": total_eligible_switching,
        "eligible_constant_fraction": (total_eligible_constant / total_eligible) if total_eligible else None,
        "eligible_switching_fraction": (total_eligible_switching / total_eligible) if total_eligible else None,
        "constant_H25_negative_objective_count": constant_negative_objective,
        "switching_negative_objective_count": switching_negative_objective,
        "seeds_with_switching_opportunity": seeds_with_switching_opportunity,
        "selected_constant_or_fixed_seeds": selected_constant_or_fixed,
        "selected_switching_seeds": selected_switching,
        "seeds_where_selected_candidate_not_best_training_objective": seeds_where_selection_not_best_training_objective,
        "seeds_where_best_switching_beats_selected_candidate": seeds_where_best_switching_beats_selected_candidate,
        "interpretation": (
            "Training populations did not collapse completely because every seed has eligible switching candidates, "
            "but selected policies/fallbacks are frequently constant-H25/fixed and timing-based objectives sometimes "
            "prefer behaviorally identical H25 candidates; this remains development evidence only."
        ),
    }

    OUT_JSON.write_text(json.dumps(digest, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Fit vs selection behavioral digest",
        "",
        f"Created UTC: {digest['created_utc']}",
        "",
        "Scope: saved vehicle training metadata only; no simulations, validation, or sealed test reads.",
        "",
        "## Cross-seed counts",
        "",
    ]
    for k, v in digest["cross_seed_findings"].items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Per-seed selected/best candidates", ""])
    for s in digest["seeds"]:
        lines.append(
            f"- seed {s['seed']}: selected={s['selected_policy']} "
            f"constant_or_fixed={s['selected_constant_or_fixed']} switching={s['selected_switching']} "
            f"best_eligible={None if s['best_eligible_by_training_objective'] is None else s['best_eligible_by_training_objective']['candidate']} "
            f"best_switching={None if s['best_switching_by_training_objective'] is None else s['best_switching_by_training_objective']['candidate']} "
            f"best_finalist={None if s['best_finalist_by_training_objective'] is None else s['best_finalist_by_training_objective']['candidate']} "
            f"flags={s['selection_vs_training_objective_flags']}"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OUT_JSON), "summary": str(OUT_MD), "cross_seed_findings": digest["cross_seed_findings"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
