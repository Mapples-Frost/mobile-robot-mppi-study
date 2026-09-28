#!/usr/bin/env python3
"""Vehicle V1 state-continuation selector smoke v0b runner.

Development-only IMPROVED selector smoke frozen by
research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0b_caselevel_gate_frozen_20260928.*

The runner tests the smallest state-level policy implied by the controlled-
continuation evidence: default H15, and latch to H30 when the online previous
state is close to one of the confirmed H30-beneficial branch states.  It compares
that selector against fixed H15/H30/H10 on the same four development cases
(5, 7, 10, 12).  It performs no training/gradient updates, opens no historical
validation64 bank, and never accesses the sealed final test.

A verified external backup proof after the case-level audit/protocol/source is
required before any rollout is allowed.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AWS_DIR = ROOT / "experiments/bohn2021_aws"
if str(AWS_DIR) not in sys.path:
    sys.path.insert(0, str(AWS_DIR))

import vehicle_fixed_h_opportunity_probe_v1_runner as v1probe  # noqa:E402

base = v1probe.base
v1 = base.v1

TASK = "vehicle"
MAX_STEPS = 150
ORDER_SEED = 2609287401
OUT_DIR = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_smoke_v0b_20260928"
STATE_PATH = ROOT / "research_artifacts/aws_state/vehicle_v1_state_continuation_selector_smoke_v0b_20260928.md"
PROTOCOL_JSON = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0b_caselevel_gate_frozen_20260928.json"
PROTOCOL_MD = ROOT / "research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0b_caselevel_gate_frozen_20260928.md"
CASELEVEL_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_caselevel_audit_v0_20260928T1250Z/completed.json"
CASELEVEL_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_caselevel_audit_v0_20260928T1250Z/raw.json"
OFFLINE_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_offline_refit_v0_20260928T1240Z/raw.json"
V1_RAW = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json"
V1_COMPLETED = ROOT / "research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json"
BACKUP_DIR = ROOT / "research_artifacts/aws_backup_proofs"
MARKER = "vehicle-v1-state-continuation-selector-smoke-v0b-20260928"


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def serial(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=serial, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=serial, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def parse_time(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def values_summary(values: Iterable[float]) -> Dict[str, Any]:
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0:
        return {"count": 0, "sum": 0.0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(xs.size), "sum": float(xs.sum()), "mean": float(xs.mean()), "median": float(np.median(xs)), "p95": float(np.percentile(xs, 95)), "max": float(xs.max())}


def verify_completed_marker(path: Path, check_hashes: bool = True) -> Dict[str, Any]:
    done = read_json(path)
    if done.get("passed") is not True:
        raise ContractError("completed marker did not pass: %s" % rel(path))
    if check_hashes:
        for name, expected in (done.get("hashes") or {}).items():
            p = ROOT / name
            if not p.exists():
                raise ContractError("completed marker references missing file: %s" % name)
            if sha256(p) != expected:
                raise ContractError("hash mismatch for completed marker input: %s" % name)
    return done


def verify_backup_proof(path: Path, min_time: Optional[dt.datetime]) -> Dict[str, Any]:
    if not path.exists():
        raise ContractError("backup proof path does not exist: %s" % rel(path))
    proof = read_json(path)
    verified = proof.get("backup_verified") is True or proof.get("status") == "verified"
    if not verified:
        raise ContractError("backup proof is not verified: %s" % rel(path))
    if int(proof.get("remaining_changed_files", -1)) != 0:
        raise ContractError("backup proof does not record remaining_changed_files=0")
    if not proof.get("commit"):
        raise ContractError("backup proof lacks commit")
    has_asset = bool(proof.get("asset_sha256") or proof.get("release_asset_sha256") or proof.get("package_sha256") or proof.get("packages_this_run"))
    if not has_asset:
        raise ContractError("backup proof lacks a verified release/package SHA")
    proof_time = None
    for key in ("time", "created_utc", "verified_utc", "backup_utc", "timestamp"):
        proof_time = parse_time(proof.get(key))
        if proof_time is not None:
            break
    if min_time is not None:
        if proof_time is None:
            raise ContractError("backup proof lacks parseable time")
        if proof_time < min_time:
            raise ContractError("backup proof predates selector-smoke case-level protocol/audit")
    return {"path": rel(path), "sha256": sha256(path), "commit": proof.get("commit"), "remaining_changed_files": proof.get("remaining_changed_files"), "time": proof_time.isoformat() if proof_time else None, "raw_status": proof.get("status"), "backup_verified": proof.get("backup_verified"), "packages_this_run": proof.get("packages_this_run")}


def assert_fresh_output_dir() -> None:
    if not OUT_DIR.exists():
        return
    completed = OUT_DIR / "completed.json"
    if completed.exists():
        verify_completed_marker(completed)
        raise SystemExit("selector smoke v0b already completed and verified; refusing rerun")
    leftovers = [p for p in OUT_DIR.iterdir() if p.name != "run.lock"]
    if leftovers:
        raise ContractError("partial selector smoke output exists; inspect/recover first: " + ", ".join(rel(p) for p in leftovers[:20]))


def verify_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    for path in (PROTOCOL_JSON, PROTOCOL_MD, CASELEVEL_COMPLETED, CASELEVEL_RAW, OFFLINE_RAW, V1_RAW, V1_COMPLETED):
        if not path.exists():
            raise ContractError("required input missing: %s" % rel(path))
    protocol = read_json(PROTOCOL_JSON)
    if protocol.get("protocol_id") != "vehicle_v1_state_continuation_selector_smoke_v0b_caselevel_gate_frozen_20260928":
        raise ContractError("unexpected selector-smoke protocol id")
    access = protocol.get("access_rules") or {}
    if access.get("development_only") is not True or access.get("historical_validation64_bank_opened") is not False or access.get("sealed_test_accessed") is not False:
        raise ContractError("selector-smoke protocol access flags are invalid")
    if access.get("requires_verified_external_backup_after_this_protocol_before_rollout") is not True:
        raise ContractError("selector-smoke protocol must require verified backup before rollout")
    budget = protocol.get("split_and_budget") or {}
    if int(budget.get("max_episodes", -1)) != 16 or int(budget.get("control_step_upper_bound", -1)) != 2400:
        raise ContractError("unexpected selector-smoke rollout budget")
    if int(budget.get("new_training_episodes", -1)) != 0 or int(budget.get("new_gradient_steps", -1)) != 0:
        raise ContractError("selector-smoke protocol unexpectedly allows training")
    if list(budget.get("rollout_cases") or []) != [5, 7, 10, 12]:
        raise ContractError("selector-smoke cases changed")
    if list(budget.get("comparators") or []) != ["selector_latch_H30", "fixed_H15", "fixed_H30", "fixed_H10"]:
        raise ContractError("selector-smoke comparator arms changed")
    case_done = verify_completed_marker(CASELEVEL_COMPLETED)
    if case_done.get("hard_pass") is not True or case_done.get("caselevel_gate_pass") is not True:
        raise ContractError("case-level selector gate did not pass")
    if case_done.get("historical_validation64_bank_opened") is not False or case_done.get("sealed_test_accessed") is not False:
        raise ContractError("case-level audit access flags invalid")
    offline = read_json(OFFLINE_RAW)
    case_raw = read_json(CASELEVEL_RAW)
    v1_done = verify_completed_marker(V1_COMPLETED, check_hashes=False)
    v1_raw = read_json(V1_RAW)
    for label, obj in (("offline_raw", offline), ("case_raw", case_raw), ("v1_raw", v1_raw), ("v1_completed", v1_done)):
        if obj.get("historical_validation64_bank_opened") is not False or obj.get("sealed_test_accessed") is not False:
            raise ContractError(label + " access flags are invalid")
    selector = protocol.get("selector") or {}
    if int(selector.get("default_horizon", -1)) != 15 or int(selector.get("trigger_horizon", -1)) != 30:
        raise ContractError("unexpected selector default/trigger horizon")
    if int(selector.get("gradient_updates", -1)) != 0 or int(selector.get("training_episodes", -1)) != 0:
        raise ContractError("selector-smoke must be zero-gradient deterministic refit")
    return protocol, case_raw, v1_raw


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def state_xyztheta_from_mapping(state: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (safe_float(state.get("x"), float("nan")), safe_float(state.get("y"), float("nan")), safe_float(state.get("theta"), float("nan")))


def state_xyztheta_from_env(env: Any) -> Tuple[float, float, float]:
    state = env.control_system.current_state
    if isinstance(state, Mapping):
        return state_xyztheta_from_mapping(state)
    return (safe_float(getattr(state, "x", float("nan")), float("nan")), safe_float(getattr(state, "y", float("nan")), float("nan")), safe_float(getattr(state, "theta", float("nan")), float("nan")))


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def state_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if not all(math.isfinite(x) for x in a + b):
        return float("inf")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5 * abs(angle_diff(a[2], b[2])))


def proto_state(row: Mapping[str, Any]) -> Tuple[float, float, float]:
    return state_xyztheta_from_mapping(row.get("state") or {})


def nearest_distance(state: Tuple[float, float, float], prototypes: Sequence[Mapping[str, Any]]) -> Tuple[float, Optional[Mapping[str, Any]]]:
    best_d = float("inf")
    best = None
    for p in prototypes:
        d = state_distance(state, proto_state(p))
        if d < best_d:
            best_d = d
            best = p
    return best_d, best


def clean_trace(trace: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = copy.deepcopy(list(trace))
    for row in cleaned:
        row.pop("timing", None)
        row.pop("decision_plus_terminal_switch_s", None)
        for attempt in (row.get("recovery") or {}).get("attempts", []):
            attempt.pop("solver_s", None)
    return cleaned


class LatchSelector(object):
    def __init__(self, selector_protocol: Mapping[str, Any]):
        self.default_h = int(selector_protocol["default_horizon"])
        self.trigger_h = int(selector_protocol["trigger_horizon"])
        self.radius = float(selector_protocol["positive_radius"])
        self.negative_veto_radius = float(selector_protocol.get("negative_veto_radius", 0.0))
        self.positive_prototypes = list(selector_protocol.get("positive_prototypes") or [])
        self.negative_prototypes = list(selector_protocol.get("negative_veto_prototypes") or [])
        self.latched = False
        self.first_trigger_step = None
        self.trigger_events = []

    def choose(self, env: Any, step: int) -> Tuple[int, Dict[str, Any]]:
        state = state_xyztheta_from_env(env)
        pos_d, pos = nearest_distance(state, self.positive_prototypes)
        neg_d, neg = nearest_distance(state, self.negative_prototypes)
        # A zero veto radius means no open negative-veto ball; exact guard states
        # are still recorded in diagnostics but do not block a positive hit.
        negative_veto = bool(self.negative_veto_radius > 0.0 and neg_d <= self.negative_veto_radius)
        trigger_now = bool((not self.latched) and pos_d <= self.radius and not negative_veto)
        if trigger_now:
            self.latched = True
            self.first_trigger_step = int(step)
        selected_h = self.trigger_h if self.latched else self.default_h
        event = {
            "kind": "nearest_positive_state_H30_latch_v0b",
            "step": int(step),
            "selected_horizon": int(selected_h),
            "latched": bool(self.latched),
            "trigger_now": bool(trigger_now),
            "first_trigger_step": self.first_trigger_step,
            "state": {"x": state[0], "y": state[1], "theta": state[2]},
            "nearest_positive_distance": float(pos_d),
            "nearest_positive_case": None if pos is None else int(pos.get("case")),
            "nearest_positive_branch_step": None if pos is None else int(pos.get("branch_step")),
            "positive_radius": float(self.radius),
            "nearest_negative_distance": float(neg_d),
            "nearest_negative_case": None if neg is None else int(neg.get("case")),
            "nearest_negative_branch_step": None if neg is None else int(neg.get("branch_step")),
            "negative_veto_radius": float(self.negative_veto_radius),
            "negative_veto": bool(negative_veto),
        }
        if trigger_now:
            self.trigger_events.append(dict(event))
        return selected_h, event


def terminal_receipt_for_horizon(receipts: Mapping[str, Any], h: int) -> Mapping[str, Any]:
    rec = receipts.get(str(h))
    if not rec:
        raise ContractError("missing terminal receipt for H%d" % h)
    return rec


def build_arms(protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"arm_id": "selector_latch_H30", "role": "selector", "default_horizon": 15, "trigger_horizon": 30, "family": "IMPROVED_state_continuation_selector_v0b"},
        {"arm_id": "fixed_H15", "role": "fixed", "horizon": 15, "family": "fixed_H15_seed0_terminal"},
        {"arm_id": "fixed_H30", "role": "fixed", "horizon": 30, "family": "fixed_H30_seed0_terminal"},
        {"arm_id": "fixed_H10", "role": "fixed", "horizon": 10, "family": "fixed_H10_seed0_terminal"},
    ]


def randomized_schedule(cases: Sequence[int], arms: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    base_rows: List[Dict[str, Any]] = []
    idx = 0
    for case_id in cases:
        for arm_index, arm in enumerate(arms):
            base_rows.append({"case": int(case_id), "arm_index": int(arm_index), "arm_id": str(arm["arm_id"]), "rollout_index": int(idx)})
            idx += 1
    rng = np.random.RandomState(ORDER_SEED)
    schedule: List[Dict[str, Any]] = []
    for exec_idx, base_idx in enumerate(rng.permutation(len(base_rows)).tolist()):
        row = dict(base_rows[int(base_idx)])
        row["execution_index"] = int(exec_idx)
        schedule.append(row)
    return schedule


def summarize_episode(trace: List[Dict[str, Any]], reset: Dict[str, Any], construction_s: float, episode_wall_s: float,
                      item: Mapping[str, Any], arm: Mapping[str, Any], selected_meta: Mapping[str, Any],
                      terminal_switches: Sequence[Mapping[str, Any]], terminal_receipts: Mapping[str, Any],
                      counts: Mapping[str, Any], logging: Any, selector_obj: Optional[LatchSelector]) -> Dict[str, Any]:
    metric = v1.case_metrics(TASK, trace)
    decision_times = [float(r["timing"]["decision_s"]) for r in trace]
    decision_gross = [float(r["timing"]["decision_gross_s"]) for r in trace]
    controller_times = [float(r["timing"]["controller_s"]) for r in trace]
    selection_times = [float(r["timing"].get("selection_s", 0.0)) for r in trace]
    logging_times = [float(r["timing"].get("logging_s", 0.0)) for r in trace]
    decision_plus_switch = [float(r.get("decision_plus_terminal_switch_s", r["timing"]["decision_s"])) for r in trace]
    solver_times: List[float] = []
    horizon_counts: Dict[str, int] = {}
    raw_counts: Dict[str, int] = {}
    trigger_steps: List[int] = []
    for row in trace:
        h = str(row["horizon"])
        horizon_counts[h] = horizon_counts.get(h, 0) + 1
        d = row.get("decision") or {}
        raw_h = str(d.get("raw_horizon", d.get("selected_horizon", row["horizon"])))
        raw_counts[raw_h] = raw_counts.get(raw_h, 0) + 1
        if bool(d.get("trigger_now")):
            trigger_steps.append(int(d.get("step", len(trigger_steps))))
        for attempt in (row.get("recovery") or {}).get("attempts", []):
            if attempt.get("solver_s") is not None:
                solver_times.append(float(attempt["solver_s"]))
    out = dict(metric)
    out.update({
        "execution_index": int(item["execution_index"]),
        "case": int(item["case"]),
        "source_candidate_index": int(selected_meta.get("candidate_index", -1)),
        "stratum": selected_meta.get("stratum"),
        "arm_id": str(arm["arm_id"]),
        "family": str(arm["family"]),
        "role": str(arm["role"]),
        "episode_failure": not bool(metric.get("success")),
        "selector_triggered": bool(selector_obj is not None and selector_obj.first_trigger_step is not None),
        "selector_first_trigger_step": None if selector_obj is None else selector_obj.first_trigger_step,
        "selector_trigger_events": [] if selector_obj is None else selector_obj.trigger_events,
        "trigger_steps_in_trace": trigger_steps,
        "trace_clean_sha256": canonical_sha(clean_trace(trace)),
        "terminal_receipts_used": {str(h): terminal_receipt_for_horizon(terminal_receipts, int(h)) for h in sorted(set(int(x) for x in horizon_counts))},
        "terminal_switches": list(terminal_switches),
        "terminal_switch_timing_s": values_summary(float(s.get("switch_s", 0.0)) for s in terminal_switches),
        "construction_s": float(construction_s),
        "episode_wall_s_including_construction_reset_tracewrites": float(episode_wall_s),
        "reset": reset,
        "decision_timing_s": values_summary(decision_times),
        "decision_gross_timing_s": values_summary(decision_gross),
        "decision_plus_terminal_switch_timing_s": values_summary(decision_plus_switch),
        "controller_timing_s_logging_deducted": values_summary(controller_times),
        "selection_timing_s": values_summary(selection_times),
        "logging_timing_s": values_summary(logging_times),
        "solver_attempt_timing_s": values_summary(solver_times),
        "deadline_exceed_steps": int(np.sum(np.asarray(decision_times, dtype=float) > 0.1)),
        "horizon_counts": horizon_counts,
        "raw_horizon_counts_before_latch": raw_counts,
        "unique_horizons": sorted(int(x) for x in horizon_counts),
        "steps_metered": int(counts.get("step_calls", 0)),
        "resets_metered": int(counts.get("reset_calls", 0)),
        "logging_operations": int(getattr(logging, "operations", 0)),
        "logging_total_s": float(getattr(logging, "seconds", 0.0)),
    })
    return out


def run_episode(item: Mapping[str, Any], arm: Mapping[str, Any], case: Mapping[str, Any], selected_meta: Mapping[str, Any],
                terminals: Mapping[int, Tuple[Any, Any]], terminal_receipts: Mapping[str, Any], selector_protocol: Mapping[str, Any]) -> Dict[str, Any]:
    ep_dir = OUT_DIR / "episodes" / ("exec%03d_case%02d_%s" % (int(item["execution_index"]), int(item["case"]), str(arm["arm_id"])))
    ep_dir.mkdir(parents=True, exist_ok=False)
    episode_start = time.perf_counter()
    construct_start = time.perf_counter()
    env = v1.make_env(TASK, 0, aligned=True, scaled_obs=True)
    counts = v1.meter(env, ep_dir)
    if arm["role"] == "fixed":
        current_terminal_h = int(arm["horizon"])
    else:
        current_terminal_h = int(selector_protocol["default_horizon"])
    env.set_value_function_weights_and_biases(*terminals[current_terminal_h])
    construction_s = time.perf_counter() - construct_start
    controller = env.control_system.controller
    original = controller.get_action
    measured: List[Dict[str, float]] = []
    trace: List[Dict[str, Any]] = []
    terminal_switches: List[Dict[str, Any]] = []
    selector_obj = LatchSelector(selector_protocol) if arm["role"] == "selector" else None
    with v1.LoggingTimer(ep_dir) as logging:
        recovery = v1.recovery_module.install(controller.mpc, logging)

        def timed(*args: Any, **kwargs: Any) -> Any:
            before = logging.seconds
            start = time.perf_counter()
            value = original(*args, **kwargs)
            gross = time.perf_counter() - start
            logged = logging.seconds - before
            if not (0.0 <= logged < gross):
                raise ContractError("invalid logging timing bounds")
            measured.append({"controller_gross_s": float(gross), "logging_s": float(logged), "controller_s": float(gross - logged)})
            return value

        controller.get_action = timed
        recovery.update(enabled=False, events=[], case=int(item["case"]), step=-1, horizon=current_terminal_h)
        reset_start = time.perf_counter()
        obs = env.reset(**copy.deepcopy(dict(case)))
        reset_gross_s = time.perf_counter() - reset_start
        if obs is None or len(measured) != 1:
            raise ContractError("unexpected reset/controller warmup state")
        reset_record = {"reset_gross_s": float(reset_gross_s)}
        reset_record.update(measured.pop())
        write_json(ep_dir / "reset.json", reset_record)
        recovery["enabled"] = True
        raw_path = ep_dir / "trace.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            for t in range(MAX_STEPS):
                select_start = time.perf_counter()
                if arm["role"] == "fixed":
                    selected_h = int(arm["horizon"])
                    decision = {"kind": "fixed_H", "step": int(t), "selected_horizon": int(selected_h), "raw_horizon": int(selected_h), "terminal_horizon": int(selected_h)}
                else:
                    selected_h, decision = selector_obj.choose(env, int(t))  # type: ignore[union-attr]
                    decision["raw_horizon"] = int(selected_h)
                selection_s = float(time.perf_counter() - select_start)
                switch_s = 0.0
                switched_terminal = False
                if int(selected_h) != int(current_terminal_h):
                    st = time.perf_counter()
                    env.set_value_function_weights_and_biases(*terminals[int(selected_h)])
                    switch_s = float(time.perf_counter() - st)
                    terminal_switches.append({"step": int(t), "from_horizon": int(current_terminal_h), "to_horizon": int(selected_h), "switch_s": switch_s})
                    current_terminal_h = int(selected_h)
                    switched_terminal = True
                recovery["step"] = int(t)
                recovery["horizon"] = int(selected_h)
                _, terminated, row = v1.observed_step(env, TASK, int(selected_h), dict(case), int(t))
                if len(measured) != 1:
                    raise ContractError("expected exactly one measured controller call")
                timing = measured.pop()
                timing.update({"selection_s": selection_s, "decision_s": float(selection_s + timing["controller_s"]), "decision_gross_s": float(selection_s + timing["controller_gross_s"]), "terminal_switch_s_before_controller": switch_s})
                decision["terminal_horizon"] = int(current_terminal_h)
                decision["terminal_switched_before_step"] = bool(switched_terminal)
                row.update({"decision": decision, "recovery": recovery["events"][-1], "timing": timing, "decision_plus_terminal_switch_s": float(timing["decision_s"] + switch_s)})
                stream.write(json.dumps(row, default=serial, allow_nan=False) + "\n")
                stream.flush()
                trace.append(row)
                if terminated:
                    break
        if not trace or not trace[-1].get("termination"):
            raise ContractError("episode did not terminate within max steps")
        v1.audit_trace(TASK, dict(case), trace)
        write_json(ep_dir / "trace.json", trace)
        summary = summarize_episode(trace, reset_record, construction_s, time.perf_counter() - episode_start, item, arm, selected_meta, terminal_switches, terminal_receipts, counts, logging, selector_obj)
        summary.update({"path": rel(ep_dir), "solver_counts": recovery["counts"]})
        write_json(ep_dir / "summary.json", summary)
    files = [p for p in ep_dir.iterdir() if p.is_file() and p.name != "completed.json"]
    write_json(ep_dir / "completed.json", {"passed": True, "hashes": {rel(p): sha256(p) for p in sorted(files)}})
    return summary


def aggregate(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    out: Dict[str, Any] = {
        "episodes": len(episodes),
        "steps": steps,
        "success_count": int(sum(1 for e in episodes if e.get("success"))),
        "episode_failure_count": int(sum(1 for e in episodes if e.get("episode_failure"))),
        "constraint_count": int(sum(1 for e in episodes if e.get("constraint"))),
        "initial_failed_steps": int(sum(int(e.get("initial_failed_steps", 0)) for e in episodes)),
        "solver_failure_steps": int(sum(int(e.get("solver_failure_steps", 0)) for e in episodes)),
        "retries": int(sum(int(e.get("retries", 0)) for e in episodes)),
        "recovered_steps": int(sum(int(e.get("recovered_steps", 0)) for e in episodes)),
        "deadline_exceed_steps": int(sum(int(e.get("deadline_exceed_steps", 0)) for e in episodes)),
        "switches": int(sum(int(e.get("switches", 0)) for e in episodes)),
        "total_cost_sum": float(math.fsum(float(e.get("total_cost", 0.0)) for e in episodes)),
        "performance_cost_sum": float(math.fsum(float(e.get("performance_cost", 0.0)) for e in episodes)),
        "constraint_cost_sum": float(math.fsum(float(e.get("constraint_cost", 0.0)) for e in episodes)),
        "physical_constraint_cost_sum": float(math.fsum(float(e.get("physical_constraint_cost", 0.0)) for e in episodes)),
        "h_penalty_sum": float(math.fsum(float(e.get("h_penalty", 0.0)) for e in episodes)),
        "decision_total_s": float(math.fsum(float(e["decision_timing_s"]["sum"]) for e in episodes)),
        "decision_plus_terminal_switch_total_s": float(math.fsum(float(e["decision_plus_terminal_switch_timing_s"]["sum"]) for e in episodes)),
        "solver_attempt_total_s": float(math.fsum(float(e["solver_attempt_timing_s"]["sum"]) for e in episodes)),
        "selection_total_s": float(math.fsum(float(e["selection_timing_s"]["sum"]) for e in episodes)),
        "terminal_switch_total_s": float(math.fsum(float(e["terminal_switch_timing_s"]["sum"]) for e in episodes)),
        "construction_total_s": float(math.fsum(float(e.get("construction_s", 0.0)) for e in episodes)),
        "reset_total_s": float(math.fsum(float((e.get("reset") or {}).get("reset_gross_s", 0.0)) for e in episodes)),
        "selector_triggered_episodes": int(sum(1 for e in episodes if e.get("selector_triggered"))),
    }
    horizons: Dict[str, int] = {}
    for e in episodes:
        for h, n in (e.get("horizon_counts") or {}).items():
            horizons[h] = horizons.get(h, 0) + int(n)
    out.update({
        "total_cost_mean_episode": out["total_cost_sum"] / len(episodes) if episodes else None,
        "physical_constraint_cost_mean_episode": out["physical_constraint_cost_sum"] / len(episodes) if episodes else None,
        "decision_mean_s_per_step": out["decision_total_s"] / steps if steps else None,
        "decision_plus_terminal_switch_mean_s_per_step": out["decision_plus_terminal_switch_total_s"] / steps if steps else None,
        "horizon_counts": horizons,
        "unique_horizons": sorted(int(x) for x in horizons),
    })
    return out


def no_safety_regression(selector: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    if bool(ref.get("success")) and not bool(selector.get("success")):
        return False
    if bool(selector.get("constraint")) and not bool(ref.get("constraint")):
        return False
    if int(selector.get("initial_failed_steps", 0)) > int(ref.get("initial_failed_steps", 0)):
        return False
    if int(selector.get("solver_failure_steps", 0)) > int(ref.get("solver_failure_steps", 0)):
        return False
    return True


def analyze(episodes: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    by_arm = {arm: aggregate([e for e in episodes if e["arm_id"] == arm]) for arm in ["selector_latch_H30", "fixed_H15", "fixed_H30", "fixed_H10"]}
    per_case: Dict[str, Any] = {}
    opportunity_improved = []
    safety_failures = []
    large_guard_regressions = []
    for case in [5, 7, 10, 12]:
        rows = {e["arm_id"]: e for e in episodes if int(e["case"]) == case}
        if set(rows) != set(["selector_latch_H30", "fixed_H15", "fixed_H30", "fixed_H10"]):
            raise ContractError("missing arm for case %d" % case)
        sel = rows["selector_latch_H30"]
        h15 = rows["fixed_H15"]
        phys_delta = float(sel["physical_constraint_cost"] - h15["physical_constraint_cost"])
        total_delta = float(sel["total_cost"] - h15["total_cost"])
        safe = no_safety_regression(sel, h15)
        if not safe:
            safety_failures.append(case)
        guard_regression = bool(case in (5, 12) and (phys_delta > max(3.0, 0.05 * abs(float(h15["physical_constraint_cost"]))) or total_delta > max(3.0, 0.05 * abs(float(h15["total_cost"])))))
        if guard_regression:
            large_guard_regressions.append(case)
        opp_improved = bool(case in (7, 10) and bool(sel.get("selector_triggered")) and safe and (phys_delta < -1e-8 or total_delta < -1e-8))
        if opp_improved:
            opportunity_improved.append(case)
        per_case[str(case)] = {
            "selector_vs_H15_physical_delta": phys_delta,
            "selector_vs_H15_total_delta": total_delta,
            "selector_vs_H15_decision_time_ratio": float(sel["decision_timing_s"]["sum"] / h15["decision_timing_s"]["sum"]) if h15["decision_timing_s"]["sum"] else None,
            "selector_triggered": bool(sel.get("selector_triggered")),
            "selector_first_trigger_step": sel.get("selector_first_trigger_step"),
            "selector_horizon_counts": sel.get("horizon_counts"),
            "fixed_H15_steps": int(h15.get("steps", 0)),
            "fixed_H30_physical_cost": float(rows["fixed_H30"].get("physical_constraint_cost", 0.0)),
            "fixed_H10_physical_cost": float(rows["fixed_H10"].get("physical_constraint_cost", 0.0)),
            "safety_no_regression_vs_H15": safe,
            "guard_large_regression": guard_regression,
            "opportunity_case_improved": opp_improved,
            "success": {arm: bool(row.get("success")) for arm, row in rows.items()},
            "termination": {arm: row.get("termination") for arm, row in rows.items()},
            "solver_failure_steps": {arm: int(row.get("solver_failure_steps", 0)) for arm, row in rows.items()},
            "initial_failed_steps": {arm: int(row.get("initial_failed_steps", 0)) for arm, row in rows.items()},
            "constraint": {arm: bool(row.get("constraint")) for arm, row in rows.items()},
        }
    safety_pass = bool(not safety_failures and by_arm["selector_latch_H30"]["constraint_count"] == 0)
    opportunity_pass = bool(len(opportunity_improved) >= 1)
    guard_pass = bool(not large_guard_regressions)
    expansion_gate_pass = bool(safety_pass and opportunity_pass and guard_pass)
    return {
        "by_arm": by_arm,
        "per_case": per_case,
        "safety_pass": safety_pass,
        "opportunity_pass": opportunity_pass,
        "guard_pass": guard_pass,
        "opportunity_improved_cases": opportunity_improved,
        "safety_failure_cases": safety_failures,
        "large_guard_regression_cases": large_guard_regressions,
        "selector_smoke_expansion_gate_pass": expansion_gate_pass,
        "acceptance_rule": protocol.get("primary_checks"),
        "interpretation": "Development smoke only. Passing authorizes a broader versioned selector/refit diagnostic, not validation/test claims.",
    }


def write_backup_request(raw: Mapping[str, Any]) -> str:
    stamp = str(raw["created_utc"]).replace("-", "").replace(":", "").replace("+00:00", "+0000")
    path = BACKUP_DIR / ("REQUEST_BACKUP_AFTER_VEHICLE_V1_STATE_CONTINUATION_SELECTOR_SMOKE_V0B_%s.json" % stamp)
    write_json(path, {
        "requested_utc": raw["created_utc"],
        "reason": "backup selector-smoke rollout before broader selector/refit/training or validation work",
        "backup_required_before_more_simulations": True,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": raw["budget_actual"]["episodes"],
        "control_steps": raw["budget_actual"]["control_steps"],
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "artifacts": [rel(OUT_DIR), rel(STATE_PATH), rel(Path(__file__).resolve()), rel(PROTOCOL_MD), rel(PROTOCOL_JSON)],
    })
    return rel(path)


def write_summary(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    lines = [
        "# Vehicle V1 state-continuation selector smoke v0b",
        "",
        f"Created UTC: `{raw['created_utc']}`.",
        "",
        "Development-only IMPROVED selector smoke: nearest confirmed H30-beneficial state latch compared with fixed H15/H30/H10. No training/gradient updates, no validation64-bank access, no sealed-test access.",
        "",
        "## Access and budget",
        "",
        f"- Backup proof: `{raw['backup_proof']['path']}` commit `{raw['backup_proof']['commit']}`.",
        f"- Episodes: `{raw['budget_actual']['episodes']}` / frozen max `{raw['budget_declared']['max_episodes']}`.",
        f"- Control steps: `{raw['budget_actual']['control_steps']}` / upper bound `{raw['budget_declared']['control_step_upper_bound']}`.",
        f"- historical_validation64_bank_opened: `{raw['historical_validation64_bank_opened']}`; sealed_test_accessed: `{raw['sealed_test_accessed']}`.",
        "",
        "## Arm aggregates",
        "",
        "| arm | episodes | steps | success | constraints | init-fail steps | solver-fail steps | physical+constraint | total | decision mean s/step | horizons | selector triggered eps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for arm in ["selector_latch_H30", "fixed_H15", "fixed_H30", "fixed_H10"]:
        a = analysis["by_arm"][arm]
        lines.append("| `%s` | %d | %d | %d | %d | %d | %d | %.6g | %.6g | %.6g | `%s` | %d |" % (
            arm, a["episodes"], a["steps"], a["success_count"], a["constraint_count"], a["initial_failed_steps"], a["solver_failure_steps"], a["physical_constraint_cost_sum"], a["total_cost_sum"], a["decision_mean_s_per_step"], a["horizon_counts"], a["selector_triggered_episodes"]
        ))
    lines += [
        "",
        "## Paired case deltas: selector minus fixed H15",
        "",
        "| case | triggered | first trigger | physical delta | total delta | decision time ratio | safety pass | opportunity improved | guard large regression | horizons |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in [5, 7, 10, 12]:
        row = analysis["per_case"][str(case)]
        ratio = row["selector_vs_H15_decision_time_ratio"]
        lines.append("| %d | `%s` | %s | %.6g | %.6g | %s | `%s` | `%s` | `%s` | `%s` |" % (
            case,
            str(row["selector_triggered"]),
            str(row["selector_first_trigger_step"]),
            float(row["selector_vs_H15_physical_delta"]),
            float(row["selector_vs_H15_total_delta"]),
            "None" if ratio is None else ("%.6g" % float(ratio)),
            str(row["safety_no_regression_vs_H15"]),
            str(row["opportunity_case_improved"]),
            str(row["guard_large_regression"]),
            row["selector_horizon_counts"],
        ))
    lines += [
        "",
        "## Gate result",
        "",
        f"- safety_pass: `{analysis['safety_pass']}`; opportunity_pass: `{analysis['opportunity_pass']}`; guard_pass: `{analysis['guard_pass']}`.",
        f"- opportunity_improved_cases: `{analysis['opportunity_improved_cases']}`.",
        f"- selector_smoke_expansion_gate_pass: `{analysis['selector_smoke_expansion_gate_pass']}`.",
        "",
    ]
    if analysis["selector_smoke_expansion_gate_pass"]:
        lines.append("Interpretation: the state-level latch can exploit at least one confirmed opportunity without smoke-level safety/guard regression; next step is a frozen broader development confirmation/refit protocol with independent refit/training seeds and strong fixed-H baselines.")
    else:
        lines.append("Interpretation: the simple nearest-state latch failed its development-smoke expansion gate; next work should diagnose representation/prototype radius/terminal mismatch before broader validation.")
    lines += ["", f"Backup request before further simulations: `{raw['backup_request']}`."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(raw: Mapping[str, Any]) -> None:
    analysis = raw["analysis"]
    block = (
        f"\n<!-- {MARKER} -->\n"
        "## 2026-09-28 vehicle V1 state-continuation selector smoke v0b\n\n"
        f"UTC: {raw['created_utc']}. Development-only selector smoke completed: {raw['budget_actual']['episodes']} episodes, "
        f"{raw['budget_actual']['control_steps']} control steps, no validation64/test access. "
        f"Expansion gate={analysis['selector_smoke_expansion_gate_pass']}; safety={analysis['safety_pass']}; "
        f"opportunity cases improved={analysis['opportunity_improved_cases']}. Artifacts: `{rel(OUT_DIR / 'summary.md')}`, `{rel(OUT_DIR / 'raw.json')}`, `{rel(OUT_DIR / 'completed.json')}`.\n"
    )
    for name in ("STATUS.md", "RESEARCH_LOG.md", "DECISIONS.md", "RESULTS_AUDIT.md", "REPRODUCTION_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if MARKER not in old:
                path.write_text(old.rstrip() + "\n" + block, encoding="utf-8")


def source_hashes() -> Dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(v1probe.__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(base.smoke_base.__file__).resolve(),
        Path(base.v1.__file__).resolve(),
        PROTOCOL_JSON,
        PROTOCOL_MD,
        CASELEVEL_COMPLETED,
        CASELEVEL_RAW,
        OFFLINE_RAW,
        V1_RAW,
        V1_COMPLETED,
        ROOT / "experiments/bohn2021_reproduction/conservative_canonical_reset.py",
        ROOT / "experiments/bohn2021_reproduction/conservative_solver_recovery.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_run.py",
        ROOT / "experiments/bohn2021_reproduction/branch_calibration_audit.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_search.py",
        ROOT / "experiments/bohn2021_reproduction/gated_horizon_timing.py",
        ROOT / "experiments/bohn2021_reproduction/run.py",
        ROOT / "experiments/bohn2021_reproduction/runtime.py",
    ]
    return {rel(p): sha256(p) for p in paths if p.exists()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup-proof", required=True, type=Path, help="verified external backup proof after the case-level selector audit/protocol/source")
    ap.add_argument("--i-accept-selector-smoke-development-rollout", action="store_true", help="explicit acknowledgement: development diagnostic only, no validation64/test access")
    args = ap.parse_args(argv)
    if not args.i_accept_selector_smoke_development_rollout:
        raise ContractError("explicit --i-accept-selector-smoke-development-rollout is required")
    assert_fresh_output_dir()
    protocol, case_raw, v1_raw = verify_inputs()
    min_backup_time = parse_time(case_raw.get("created_utc")) or parse_time(protocol.get("created_utc"))
    backup = verify_backup_proof(args.backup_proof, min_backup_time)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(OUT_DIR / "run_started.json", {"started_utc": started, "pid": os.getpid(), "method": "vehicle_v1_state_continuation_selector_smoke_v0b", "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "training_gradient_steps": 0})
    preflight = base.runtime_preflight()
    write_json(OUT_DIR / "runtime_preflight.json", preflight)
    if not preflight.get("passed"):
        raise ContractError(str(preflight.get("diagnosis", "runtime preflight failed")) + " " + str(preflight.get("exception", "")))
    v1.latency_verify()
    bank_path = ROOT / str(v1_raw["bank"]["path"])
    bank_done = bank_path.parent / "completed.json"
    verify_completed_marker(bank_done)
    bank = read_json(bank_path)
    selected_cases = bank["selected_cases"]
    selected_meta = bank["selection"]["selected_metadata"]
    rollout_cases = [int(x) for x in (protocol.get("split_and_budget") or {}).get("rollout_cases")]
    if len(selected_cases) <= max(rollout_cases):
        raise ContractError("protocol references unavailable V1 selected case")
    terminals, terminal_receipts = base.load_terminal_grid(v1_raw["protocol_full"])
    write_json(OUT_DIR / "terminal_sources.json", terminal_receipts)
    arms = build_arms(protocol)
    schedule = randomized_schedule(rollout_cases, arms)
    write_json(OUT_DIR / "schedule.json", {"order_seed": ORDER_SEED, "episodes": schedule, "arms": arms, "rollout_cases": rollout_cases})
    episodes: List[Dict[str, Any]] = []
    selector_protocol = protocol["selector"]
    arm_by_index = {i: arm for i, arm in enumerate(arms)}
    for item in schedule:
        arm = arm_by_index[int(item["arm_index"])]
        case_id = int(item["case"])
        summary = run_episode(item, arm, selected_cases[case_id], selected_meta[case_id], terminals, terminal_receipts, selector_protocol)
        episodes.append(summary)
        progress = {
            "pid": os.getpid(),
            "episodes_done": len(episodes),
            "episodes_expected": len(schedule),
            "control_steps_done": int(sum(int(e["steps"]) for e in episodes)),
            "last_episode": {k: summary[k] for k in ("execution_index", "case", "arm_id", "steps", "success", "termination")},
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
        }
        write_json(OUT_DIR / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    control_steps = int(sum(int(e.get("steps", 0)) for e in episodes))
    budget = protocol["split_and_budget"]
    if len(episodes) != int(budget["max_episodes"]) or control_steps > int(budget["control_step_upper_bound"]):
        raise ContractError("selector-smoke budget violation")
    analysis = analyze(episodes, protocol)
    raw: Dict[str, Any] = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_utc": started,
        "method": "vehicle_v1_state_continuation_selector_smoke_v0b_nearest_state_H30_latch",
        "classification": "development_IMPROVED_selector_refit_smoke_not_validation_not_final_test",
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "sealed_test_bank_opened": False,
        "backup_proof": backup,
        "protocol": {"path": rel(PROTOCOL_MD), "sha256": sha256(PROTOCOL_MD), "json_path": rel(PROTOCOL_JSON), "json_sha256": sha256(PROTOCOL_JSON)},
        "caselevel_audit": {"completed": rel(CASELEVEL_COMPLETED), "completed_sha256": sha256(CASELEVEL_COMPLETED), "raw": rel(CASELEVEL_RAW), "raw_sha256": sha256(CASELEVEL_RAW), "headline": {"caselevel_gate_pass": True}},
        "offline_refit": {"raw": rel(OFFLINE_RAW), "raw_sha256": sha256(OFFLINE_RAW)},
        "v1_source": {"raw": rel(V1_RAW), "raw_sha256": sha256(V1_RAW), "completed": rel(V1_COMPLETED), "completed_sha256": sha256(V1_COMPLETED)},
        "bank": {"path": rel(bank_path), "sha256": sha256(bank_path)},
        "source_hashes": source_hashes(),
        "platform": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "thread_environment": {k: v for k, v in os.environ.items() if k.endswith("NUM_THREADS") or k.startswith("TF_NUM_")}},
        "runtime_preflight": preflight,
        "budget_declared": {"max_episodes": int(budget["max_episodes"]), "control_step_upper_bound": int(budget["control_step_upper_bound"]), "new_training_episodes": 0, "new_gradient_steps": 0},
        "budget_actual": {"episodes": len(episodes), "control_steps": control_steps, "environment_constructions": len(episodes), "episode_resets": int(sum(int(e.get("resets_metered", 0)) for e in episodes)), "new_training_episodes": 0, "new_gradient_steps": 0, "historical_validation64_episodes": 0, "sealed_test_episodes": 0},
        "schedule": schedule,
        "terminal_sources": terminal_receipts,
        "episodes": episodes,
        "analysis": analysis,
        "interpretation_limits": [
            "development smoke only",
            "selector prototypes/radius derived from already-opened controlled-continuation development traces",
            "not validation or final-test evidence",
            "not ORIGINAL SAC and not gradient RL training",
            "actual timing from AWS run; selection/terminal-switch overhead reported separately",
        ],
    }
    raw["backup_request"] = write_backup_request(raw)
    write_json(OUT_DIR / "raw.json", raw)
    write_summary(raw)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        f"# Vehicle V1 state-continuation selector smoke v0b state ({raw['created_utc']})\n\n"
        f"Completed {len(episodes)} episodes / {control_steps} control steps. No validation64/test access. "
        f"Expansion gate={analysis['selector_smoke_expansion_gate_pass']}; safety={analysis['safety_pass']}; "
        f"opportunity_improved_cases={analysis['opportunity_improved_cases']}. Backup required before further simulation.\n",
        encoding="utf-8",
    )
    append_docs(raw)
    files = [p for p in OUT_DIR.rglob("*") if p.is_file() and p.name != "completed.json"] + [ROOT / raw["backup_request"], STATE_PATH, Path(__file__).resolve(), PROTOCOL_MD, PROTOCOL_JSON, CASELEVEL_COMPLETED, CASELEVEL_RAW]
    write_json(OUT_DIR / "completed.json", {
        "passed": True,
        "hard_pass": True,
        "formal_scientific_evidence": False,
        "historical_validation64_bank_opened": False,
        "sealed_test_accessed": False,
        "episodes": len(episodes),
        "control_steps": control_steps,
        "new_training_episodes": 0,
        "new_gradient_steps": 0,
        "backup_request": raw["backup_request"],
        "headline": {"selector_smoke_expansion_gate_pass": analysis["selector_smoke_expansion_gate_pass"], "safety_pass": analysis["safety_pass"], "opportunity_pass": analysis["opportunity_pass"], "opportunity_improved_cases": analysis["opportunity_improved_cases"]},
        "hashes": {rel(p): sha256(p) for p in sorted(set(files)) if p.exists()},
    })
    print(json.dumps({"completed": rel(OUT_DIR / "completed.json"), "summary": rel(OUT_DIR / "summary.md"), "episodes": len(episodes), "control_steps": control_steps, "selector_smoke_expansion_gate_pass": analysis["selector_smoke_expansion_gate_pass"], "safety_pass": analysis["safety_pass"], "opportunity_pass": analysis["opportunity_pass"], "historical_validation64_bank_opened": False, "sealed_test_accessed": False, "backup_request": raw["backup_request"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "failure.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "historical_validation64_bank_opened": False,
            "sealed_test_accessed": False,
            "new_training_episodes": 0,
            "new_gradient_steps": 0,
            "next_recovery_hint": "Preserve partial directory, audit failure, and do not broaden changes without a new hypothesis. Use legacy interpreter and a verified backup proof after the selector case-level audit/protocol/source.",
        })
        raise
