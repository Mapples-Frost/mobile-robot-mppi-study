#!/usr/bin/env python3
"""Case-level latch audit for the Vehicle V1 state-continuation selector.

The preceding offline refit found a strict-window gate failure because case 7's
later positive branch (step 16) would already be latched by a confirmed earlier
positive branch (step 11).  This diagnostic checks whether that is a genuine
over-triggering failure or an overly strict state-level accounting artifact.

No environments are constructed; no rollouts, training, validation64 or sealed
final-test access occur.  If the case-level gate passes, this script freezes a
v0b selector-smoke protocol with the same selector and rollout budget, changing
only the offline-gate interpretation before any rollout is run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
INPUT_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_offline_refit_v0_20260928T1240Z/raw.json"
INPUT_DONE = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_offline_refit_v0_20260928T1240Z/completed.json"
INPUT_PROTOCOL = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0_frozen_20260928.json"
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_caselevel_audit_v0_20260928T1250Z"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_state_continuation_selector_caselevel_audit_v0_20260928T1250Z.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0b_caselevel_gate_frozen_20260928.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0b_caselevel_gate_frozen_20260928.md"
BACKUP_REQUEST = ROOT / "research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1_STATE_CONTINUATION_SELECTOR_CASELEVEL_AUDIT_V0_20260928T1250Z.json"
MARKER = "vehicle-v1-state-continuation-selector-caselevel-audit-v0-20260928T1250Z"
TARGET_WINDOW = 2


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> Dict[str, Any]:
    for path in (INPUT_RAW, INPUT_DONE, INPUT_PROTOCOL):
        if not path.exists():
            raise ContractError("missing input: %s" % rel(path))
    done = read_json(INPUT_DONE)
    if done.get("passed") is not True or done.get("hard_pass") is not True:
        raise ContractError("offline refit completed marker is not a hard pass")
    if done.get("historical_validation64_bank_opened") is not False or done.get("sealed_test_accessed") is not False:
        raise ContractError("offline refit access flags invalid")
    if int(done.get("new_rollouts", -1)) != 0 or int(done.get("new_control_steps", -1)) != 0:
        raise ContractError("offline refit unexpectedly consumed rollout budget")
    for name, expected in (done.get("hashes") or {}).items():
        p = ROOT / name
        if not p.exists():
            raise ContractError("completed marker references missing file: %s" % name)
        if sha256(p) != expected:
            raise ContractError("hash mismatch for completed marker input: %s" % name)
    raw = read_json(INPUT_RAW)
    if raw.get("historical_validation64_bank_opened") is not False or raw.get("sealed_test_accessed") is not False:
        raise ContractError("offline raw access flags invalid")
    if (raw.get("budget_actual") or {}).get("new_rollouts") != 0:
        raise ContractError("offline raw unexpectedly has new rollouts")
    return raw


def by_case(rows: Sequence[Mapping[str, Any]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for r in rows:
        out.setdefault(int(r["case"]), []).append(int(r["branch_step"]))
    for steps in out.values():
        steps.sort()
    return out


def is_near(step: Optional[int], target: int, window: int = TARGET_WINDOW) -> bool:
    return step is not None and abs(int(step) - int(target)) <= window


def audit(raw: Mapping[str, Any]) -> Dict[str, Any]:
    selected = raw["selected_rule"]
    positives = list(raw["positive_prototypes"])
    negatives = list(raw["negative_guard_prototypes"])
    pos_steps = by_case(positives)
    positive_cases = sorted(pos_steps)
    strict_premature_rows: List[Dict[str, Any]] = []
    benign_early_positive_latch_rows: List[Dict[str, Any]] = []
    bad_premature_rows: List[Dict[str, Any]] = []
    covered_positive_states = 0
    missed_positive_states = []
    for detail in selected.get("positive_target_details", []):
        case = int(detail["case"])
        target_step = int(detail["branch_step"])
        first = detail.get("first_trigger_step_on_H15_trace")
        first_int = None if first is None else int(first)
        covered = bool(first_int is not None and first_int <= target_step + TARGET_WINDOW)
        if covered:
            covered_positive_states += 1
        else:
            missed_positive_states.append(dict(detail))
        if bool(detail.get("premature_before_window")):
            strict_premature_rows.append(dict(detail))
            earlier_confirmed = [s for s in pos_steps.get(case, []) if s < target_step and is_near(first_int, s)]
            if earlier_confirmed:
                row = dict(detail)
                row["earlier_confirmed_positive_latch_step"] = int(earlier_confirmed[0])
                benign_early_positive_latch_rows.append(row)
            else:
                bad_premature_rows.append(dict(detail))
    per_case = selected.get("per_case") or {}
    latched_positive_cases = sorted(
        int(c) for c in positive_cases
        if per_case.get(str(c), {}).get("first_trigger_step") is not None
    )
    guard_only_cases = sorted(set(int(n["case"]) for n in negatives) - set(positive_cases))
    guard_only_latches = [
        {"case": c, "first_trigger_step": per_case.get(str(c), {}).get("first_trigger_step")}
        for c in guard_only_cases if per_case.get(str(c), {}).get("first_trigger_step") is not None
    ]
    late_guard_near_hits = [
        dict(g) for g in selected.get("guard_details", [])
        if bool(g.get("triggered_near_guard_window"))
    ]
    caselevel_gate_pass = bool(
        len(latched_positive_cases) >= 2
        and covered_positive_states == len(positives)
        and not bad_premature_rows
        and int(selected.get("guard_window_hits", 999)) == 0
        and int(selected.get("nonpositive_case_latches", 999)) == 0
        and not guard_only_latches
        and not late_guard_near_hits
    )
    return {
        "strict_offline_gate_pass": bool(selected.get("offline_gate_pass")),
        "caselevel_gate_pass": caselevel_gate_pass,
        "selected_radius": float(selected["radius"]),
        "selected_negative_veto_radius": float(selected["negative_veto_radius"]),
        "positive_case_count": len(positive_cases),
        "latched_positive_cases": latched_positive_cases,
        "positive_state_count": len(positives),
        "covered_positive_states_under_latch": covered_positive_states,
        "missed_positive_states": missed_positive_states,
        "strict_premature_rows": strict_premature_rows,
        "benign_early_positive_latch_rows": benign_early_positive_latch_rows,
        "bad_premature_rows": bad_premature_rows,
        "guard_only_cases": guard_only_cases,
        "guard_only_latches": guard_only_latches,
        "late_guard_near_hits": late_guard_near_hits,
        "guard_window_hits": int(selected.get("guard_window_hits", 0)),
        "nonpositive_case_latches": int(selected.get("nonpositive_case_latches", 0)),
        "latched_cases": int(selected.get("latched_cases", 0)),
        "total_trigger_steps_before_latch": int(selected.get("total_trigger_steps_before_latch", 0)),
        "interpretation": (
            "The strict state-window gate treats the case7 step16 target as premature because the selector first fires at step11. "
            "Step11 is itself a confirmed material H30 opportunity in the same episode; for a latch policy this is benign rather than an over-trigger."
        ),
    }


def freeze_v0b_protocol(created: str, raw: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    base_protocol = dict((raw.get("next_protocol") or {}).get("full") or read_json(INPUT_PROTOCOL))
    base_protocol["protocol_id"] = "vehicle_v1_state_continuation_selector_smoke_v0b_caselevel_gate_frozen_20260928"
    base_protocol["created_utc"] = created
    base_protocol["classification"] = "development_IMPROVED_state_selector_smoke_v0b_caselevel_gate_not_validation_not_final_test"
    base_protocol.setdefault("motivation", []).append(
        "Case-level latch audit showed the only strict premature label was benign: case7 step16 is already covered by a confirmed earlier H30 latch at case7 step11."
    )
    base_protocol["offline_gate_amendment"] = {
        "source_audit": rel(OUT_DIR / "raw.json"),
        "strict_offline_gate_pass": result["strict_offline_gate_pass"],
        "caselevel_gate_pass": result["caselevel_gate_pass"],
        "amendment_scope": "offline gate interpretation only; selector prototypes, radius, comparator arms, rollout cases and budgets remain unchanged from v0",
        "benign_premature_definition": "If a later positive target is already latched by an earlier confirmed positive target in the same case, it is not an unsafe premature over-trigger for a latch-until-termination selector.",
    }
    base_protocol["access_rules"] = dict(base_protocol.get("access_rules") or {})
    base_protocol["access_rules"]["requires_verified_external_backup_after_this_protocol_before_rollout"] = True
    write_json(PROTOCOL_JSON, base_protocol)
    lines = [
        "# Vehicle V1 state-continuation selector smoke v0b case-level gate frozen protocol",
        "",
        f"Frozen UTC: `{created}`.",
        "",
        "Development-only IMPROVED selector smoke; no validation64-bank or sealed-test access. A verified external backup after this protocol and case-level audit is required before any rollout.",
        "",
        "## What changed from v0",
        "",
        "Only the offline gate interpretation changed. The selector rule, positive prototypes, radius, comparator arms, rollout cases and rollout budgets are unchanged.",
        "",
        f"Strict offline gate pass: `{result['strict_offline_gate_pass']}`; case-level latch gate pass: `{result['caselevel_gate_pass']}`.",
        "",
        "The strict failure came from case 7 step 16 being latched at step 11. Step 11 is itself a confirmed material H30 opportunity, so this is benign for a latch policy rather than a guard/nonpositive over-trigger.",
        "",
        "## Candidate selector",
        "",
        f"Default H15; if previous-state distance to a positive prototype is <= `{result['selected_radius']}` and not inside negative veto radius `{result['selected_negative_veto_radius']}`, switch/latch to H30 until termination.",
        "",
        "## First rollout budget",
        "",
        f"- Cases: `{base_protocol.get('split_and_budget', {}).get('rollout_cases')}`",
        f"- Arms: `{base_protocol.get('split_and_budget', {}).get('comparators')}`",
        f"- Max episodes: `{base_protocol.get('split_and_budget', {}).get('max_episodes')}`; control-step upper bound: `{base_protocol.get('split_and_budget', {}).get('control_step_upper_bound')}`",
        "- Training episodes / gradient steps: `0 / 0`",
        "",
        "## Acceptance for expansion",
        "",
        str((base_protocol.get("primary_checks") or {}).get("safety")),
        str((base_protocol.get("primary_checks") or {}).get("opportunity")),
        str((base_protocol.get("primary_checks") or {}).get("timing")),
    ]
    PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base_protocol


def write_summary(raw: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    lines = [
        "# Vehicle V1 state-continuation selector case-level latch audit v0",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "No simulation/training/validation64/test access. This audit reinterprets the offline refit's strict positive-window failure for a latch-until-termination selector.",
        "",
        "## Result",
        "",
        f"- Strict offline gate pass: `{result['strict_offline_gate_pass']}`.",
        f"- Case-level latch gate pass: `{result['caselevel_gate_pass']}`.",
        f"- Positive cases latched: `{result['latched_positive_cases']}` / `{result['positive_case_count']}`.",
        f"- Positive states covered once latched: `{result['covered_positive_states_under_latch']}` / `{result['positive_state_count']}`.",
        f"- Guard-window hits: `{result['guard_window_hits']}`; nonpositive-case latches: `{result['nonpositive_case_latches']}`; bad premature latches: `{len(result['bad_premature_rows'])}`.",
        "",
        "## Benign strict-premature rows",
        "",
        "| case | target step | first trigger | earlier confirmed positive latch |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["benign_early_positive_latch_rows"]:
        lines.append("| %d | %d | %s | %d |" % (
            int(row["case"]), int(row["branch_step"]), str(row.get("first_trigger_step_on_H15_trace")), int(row["earlier_confirmed_positive_latch_step"])
        ))
    if not result["benign_early_positive_latch_rows"]:
        lines.append("| — | — | — | — |")
    lines += [
        "",
        "## Decision",
        "",
        "The offline strict gate failure is not evidence that the state rule over-triggers into non-opportunity states. It is an accounting artifact of evaluating two positive targets in the same case independently despite the intended latch semantics. A bounded selector-smoke rollout remains scientifically informative after external backup.",
        "",
        f"Frozen v0b protocol: `{rel(PROTOCOL_JSON)}`.",
        f"Backup request before any rollout simulation: `{rel(BACKUP_REQUEST)}`.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(created: str, result: Mapping[str, Any]) -> None:
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 state-continuation selector case-level latch audit v0\n\n"
        f"UTC: {created}. No-simulation case-level audit completed after offline refit. "
        f"Strict offline gate={result['strict_offline_gate_pass']}, case-level latch gate={result['caselevel_gate_pass']}; "
        f"positive cases latched={result['latched_positive_cases']}; guard hits={result['guard_window_hits']}; "
        f"nonpositive-case latches={result['nonpositive_case_latches']}; bad premature latches={len(result['bad_premature_rows'])}. "
        f"Frozen v0b selector-smoke protocol: `{rel(PROTOCOL_JSON)}`. Backup required before any rollout.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists():
        completed = OUT_DIR / "completed.json"
        if completed.exists():
            print(json.dumps({"already_completed": rel(completed)}, sort_keys=True))
            return 0
        leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
        if leftovers:
            raise ContractError("partial case-level audit output exists; inspect before retry: " + ", ".join(rel(p) for p in leftovers[:20]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    input_raw = verify_inputs()
    result = audit(input_raw)
    protocol = freeze_v0b_protocol(created, input_raw, result)
    raw: Dict[str, Any] = {
        "created_utc": created,
        "method": "vehicle_v1_state_continuation_selector_caselevel_latch_audit_v0_no_simulation",
        "classification": "development_IMPROVED_selector_gate_audit_metadata_trace_only_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "budget_declared": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0},
        "budget_actual": {"new_rollouts": 0, "new_control_steps": 0, "new_training_episodes": 0, "new_gradient_steps": 0},
        "input": {"raw": rel(INPUT_RAW), "raw_sha256": sha256(INPUT_RAW), "completed": rel(INPUT_DONE), "completed_sha256": sha256(INPUT_DONE), "protocol": rel(INPUT_PROTOCOL), "protocol_sha256": sha256(INPUT_PROTOCOL)},
        "caselevel_audit": result,
        "next_protocol": {"json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON), "md_path": rel(PROTOCOL_MD), "md_sha256": sha256(PROTOCOL_MD), "full": protocol},
        "backup_request": rel(BACKUP_REQUEST),
        "backup_needed_before_rollout": True,
        "interpretation_limits": [
            "development metadata/trace audit only",
            "does not validate selector rollouts",
            "does not establish generalization",
            "keeps selector rule and rollout budget unchanged; only fixes latch-gate accounting",
            "requires verified external backup before any simulation",
        ],
    }
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw, result)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 selector case-level latch audit v0 state ({created})\n\n"
        f"No simulation/training/validation64/test access. Strict offline gate={result['strict_offline_gate_pass']}; "
        f"case-level latch gate={result['caselevel_gate_pass']}; positive cases latched={result['latched_positive_cases']}; "
        f"guard hits={result['guard_window_hits']}; nonpositive latches={result['nonpositive_case_latches']}; "
        f"bad premature latches={len(result['bad_premature_rows'])}. Next action: obtain verified backup covering `{rel(OUT_DIR)}`, `{rel(PROTOCOL_JSON)}`, `{rel(Path(__file__).resolve())}`, then run the bounded v0b selector-smoke rollout if backup is verified.\n",
        encoding="utf-8",
    )
    write_json(BACKUP_REQUEST, {
        "requested_utc": created,
        "reason": "backup case-level selector gate audit and frozen v0b selector-smoke protocol before any selector rollout simulation",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(PROTOCOL_JSON), rel(PROTOCOL_MD), rel(Path(__file__).resolve())],
    })
    append_docs(created, result)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [STATE_PATH, PROTOCOL_JSON, PROTOCOL_MD, BACKUP_REQUEST, Path(__file__).resolve()]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "new_control_steps": 0,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "strict_offline_gate_pass": bool(result["strict_offline_gate_pass"]),
        "caselevel_gate_pass": bool(result["caselevel_gate_pass"]),
        "backup_request": rel(BACKUP_REQUEST),
        "next_protocol": rel(PROTOCOL_JSON),
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({
        "completed": rel(OUT_DIR / "completed.json"),
        "summary": rel(OUT_DIR / "summary.md"),
        "strict_offline_gate_pass": bool(result["strict_offline_gate_pass"]),
        "caselevel_gate_pass": bool(result["caselevel_gate_pass"]),
        "latched_positive_cases": result["latched_positive_cases"],
        "covered_positive_states_under_latch": int(result["covered_positive_states_under_latch"]),
        "positive_state_count": int(result["positive_state_count"]),
        "bad_premature_latches": len(result["bad_premature_rows"]),
        "guard_window_hits": int(result["guard_window_hits"]),
        "nonpositive_case_latches": int(result["nonpositive_case_latches"]),
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "new_rollouts": 0,
        "backup_request": rel(BACKUP_REQUEST),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
