#!/usr/bin/env python3
"""Vehicle stress-scenario Stage2 identical-state continuation runner v0b.

One-variable repair of v0 after the no-simulation prepare path failed before any
rollout/training/test access with KeyError('budget') while formatting the
protocol markdown.  Scientific design is unchanged: prepare-only freezes the
same Stage2 target-selection rule from Stage1 H15 traces; run-stage2 (after a
new backup) executes the same 12 targets x 6 branch horizons matched
continuation diagnostic.

Compared with v0:
- versioned output stamp is advanced to 20260928T1748Z to preserve the failed v0
  prepare directory separately;
- markdown summary reads rollout budget from protocol['rollout_design_after_backup']
  instead of the nonexistent protocol['budget'];
- provenance hashes include both this wrapper and the archived v0 source.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_stress_scenario_stage2_continuation_v0_runner as base  # noqa:E402

BASE_SCRIPT = (AWS_DIR / "vehicle_stress_scenario_stage2_continuation_v0_runner.py").resolve()
THIS_SCRIPT = Path(__file__).resolve()

# Versioned repaired outputs.  Keep the failed v0 prepare artifacts intact.
base.PROTOCOL_STAMP = "20260928T1748Z"
base.MARKER_PREPARE = "vehicle-stress-scenario-stage2-continuation-v0b-prepare-20260928T1748Z"
base.MARKER_RUN = "vehicle-stress-scenario-stage2-continuation-v0b-run-20260928T1748Z"
base.PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_stage2_continuation_v0b_frozen_20260928T1748Z.json"
base.PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_stress_scenario_stage2_continuation_v0b_frozen_20260928T1748Z.md"
base.PREPARE_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_prepare_20260928T1748Z"
base.RUN_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z"
base.STATE_PREPARE = ROOT / "research_artifacts/aws_state/vehicle_stress_scenario_stage2_continuation_v0b_prepare_20260928T1748Z.md"
base.STATE_RUN = ROOT / "research_artifacts/aws_state/vehicle_stress_scenario_stage2_continuation_v0b_run_20260928T1748Z.md"
base.REQUEST_BACKUP_BEFORE_RUN = base.BACKUP_DIR / "REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_SCENARIO_STAGE2_CONTINUATION_V0B_ROLLOUT_20260928T1748Z.json"
# Ensure base functions using their global __file__ report this repaired script as
# the active runner while the provenance hash list still records the failed v0.
base.__file__ = str(THIS_SCRIPT)


def source_hashes(extra: Sequence[Path] = ()) -> dict[str, str]:
    paths = [
        THIS_SCRIPT,
        BASE_SCRIPT,
        base.STAGE1_RAW,
        base.STAGE1_COMPLETED,
        base.STAGE1_POST_COMPLETED,
        base.STAGE1_POST_RAW,
        base.STAGE1_BANK,
        base.STAGE1_PROTOCOL_JSON,
        base.STAGE1_PROTOCOL_MD,
        base.PROTOCOL_JSON,
        base.PROTOCOL_MD,
        base.SUPERVISOR_PROOF_PATH,
    ] + list(extra)
    return {base.rel(p): base.sha256(p) for p in paths if p.exists()}


def fixed_write_protocol_and_summary(raw: Mapping[str, Any]) -> None:
    protocol = raw["protocol_full"]
    budget = protocol["rollout_design_after_backup"]
    base.write_json(base.PROTOCOL_JSON, protocol)
    lines = [
        "# Vehicle stress-scenario Stage2 identical-state continuation v0b protocol",
        "",
        f"Frozen UTC: `{raw['created_utc']}`. Development-only IMPROVED diagnostic; not validation, not model selection, not final test.",
        "",
        "## Repair provenance",
        "",
        "This v0b protocol is a one-variable implementation repair of v0 prepare-only formatting. The failed v0 run produced no simulations/training/test access; v0b preserves the same target-selection rule, horizons, budgets and analysis gates while using a new output stamp.",
        "",
        "## Hypothesis",
        "",
        "Stage1 stress cases show material episode-level fixed-H opportunity concentrated in case 5. Stage2 tests whether this is a reusable within-episode state-dependent horizon signal or merely a case-level fixed-H effect.",
        "",
        "## Fixed design",
        "",
        f"- Prefix horizon: H{base.PREFIX_H}; branch horizons: `{base.BRANCH_HORIZONS}`.",
        f"- Targets: `{len(protocol['targets'])}`; rollout episodes if executed: `{budget['rollout_episodes_exact']}`; control-step cap: `{budget['control_step_upper_bound']}`.",
        "- Target selection uses Stage1 H15 traces and metadata only for branch-step selection. No Stage2 branch outcomes exist at freeze time.",
        "- Run is blocked until an external backup covers this repaired runner, archived v0 source, frozen protocol, prepare outputs and backup request.",
        "",
        "## Targets",
        "",
        "| target | case | role | branch step | H15 trace len | score | theta_r | traj_steps | clearance |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t in protocol["targets"]:
        meta = t["case_metadata"]
        lines.append("| %d | %d | `%s` | %d | %d | %.6g | %.6g | %d | %.6g |" % (
            int(t["target_index"]), int(t["case"]), t["selection_role"], int(t["branch_step"]), int(t["h15_trace_length"]),
            float(t["h15_only_selection_features"]["transient_score_from_H15_only"]),
            float(meta.get("theta_r", float("nan"))), int(meta.get("traj_steps", 0)), float(meta.get("min_reference_obstacle_clearance", float("nan"))),
        ))
    lines += [
        "",
        "## Stage2 analysis gates (frozen before rollout)",
        "",
        "- Material positive state: non-H15 branch has identical clean H15 prefix, branch-state distance <=1e-5, no success/constraint/initial/final solver regression versus the H15 branch, and improves continuation physical or total cost by >=3 absolute units.",
        "- Training/refit label gate: >=2 material non-H15 positive matched states, at least one in the Stage1 material stress case, and >=2 retained negative/control matched states, with no blocking prefix/state/safety artifact.",
        "- If the gate fails, preserve as evidence of sparse within-episode opportunity and pivot to terminal/reward/modeling or a separately versioned stronger scenario design rather than retraining on sparse labels.",
        "",
        f"Backup request before Stage2 rollout: `{base.rel(base.REQUEST_BACKUP_BEFORE_RUN)}`.",
    ]
    base.PROTOCOL_MD.parent.mkdir(parents=True, exist_ok=True)
    base.PROTOCOL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


base.source_hashes = source_hashes
base.write_protocol_and_summary = fixed_write_protocol_and_summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except BaseException as exc:
        target = base.PREPARE_DIR if "--prepare-only" in sys.argv else base.RUN_DIR
        target.mkdir(parents=True, exist_ok=True)
        base.write_json(target / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_rollouts": 0 if "--prepare-only" in sys.argv else None,
            "new_control_steps": 0 if "--prepare-only" in sys.argv else None,
            "repair_provenance": "v0b wrapper after v0 prepare-only KeyError('budget'); preserve both sources and failed v0 artifacts",
            "next_recovery_hint": "Preserve partial output. If prepare failed, repair target/protocol preparation only; if rollout failed, audit partial episodes before rerun.",
        })
        raise
