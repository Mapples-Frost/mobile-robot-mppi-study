#!/usr/bin/env python3
"""Run the complete B11 stack on live sensors without vehicle motion output."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for value in (PROJECT_ROOT, PROJECT_ROOT / "src", HERE):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_pi5_full_config import (  # noqa: E402
    build_pi5_full_config,
    enable_single_final_control_authority,
)
from mobile_robot_mppi.core.types import (  # noqa: E402
    Pose2D,
    RobotObservation,
    Twist2D,
)
from mobile_robot_mppi.real_robot import (  # noqa: E402
    LivoxPointCloudFrame,
    LivoxScanAdapter,
    LivoxScanAdapterConfig,
    LivoxUdpReceiver,
    MaplessStaticDynamicFilter,
    MotionBootstrapMultiObstacleTracker,
    ScoutGuardedCanGateway,
    ScoutZeroOnlyCanGuard,
)
from mobile_robot_mppi.runtime.factories import make_components  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timing(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, int(np.ceil(0.95 * len(values))) - 1)],
        "max_ms": max(values),
    }


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_adapter_config(path):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Livox adapter config root must be a mapping")
    allowed = set(LivoxScanAdapterConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown Livox adapter fields: %s" % ", ".join(unknown))
    if "lidar_to_base" in values:
        matrix = np.asarray(values["lidar_to_base"], dtype=np.float64)
        values["lidar_to_base"] = tuple(matrix.reshape(-1).tolist())
    return LivoxScanAdapterConfig(**values)


def _install_mapless_tracker(perception, human_leg_mode=False):
    base = perception.dynamic_obstacle_tracker
    if base is None:
        raise RuntimeError("Full Proposed did not construct its dynamic tracker")
    overrides = {}
    if human_leg_mode:
        # Real-person profile derived from the 2026-08-03 physical trace.  It
        # admits compact, coherently translating leg clusters while retaining
        # the association, displacement, residual, and support-beam gates that
        # rejected every observed static/background track in the offline audit.
        person_tracking = dict(
            getattr(perception, "config", {}).get("person_tracking", {})
            or {}
        )
        person_gate = dict(person_tracking.get("forecast_gate", {}) or {})
        point_cloud_gate = dict(person_tracking.get("point_cloud", {}) or {})
        overrides = {
            "allow_compact_dynamic": True,
            "minimum_duration_s": 0.40,
            "dynamic_speed_mps": 0.20,
            "dynamic_displacement_m": 0.14,
            "minimum_direction_coherence": 0.60,
            "maximum_fit_residual_m": 0.09,
            "maximum_step_m": 0.35,
            "persistent_minimum_tracker_speed_mps": 0.15,
            "minimum_dynamic_support_beams": 3,
            # Mid-360 body/leg fragmentation made the same person alternate
            # between wide and one-beam clusters in the 041158 physical run.
            # Keep the early wide observation as causal shape evidence while
            # still requiring smooth displacement and a low fit residual.
            "allow_recent_vehicle_fragment_dynamic": True,
            "recent_vehicle_minimum_direction_coherence": 0.45,
            # Do not hide an otherwise valid CA-IMM forecast after only two
            # fragmented measurements.  This is a label/forecast continuity
            # hold; it does not replay a control command.
            "dynamic_hold_cycles": 12,
            "dynamic_memory_ttl_s": 1.50,
            "dynamic_handoff_gate_m": 0.50,
            # Release slots captured by stationary walls/table legs after the
            # causal static classifier agrees on three associated updates.
            "static_track_eviction_cycles": 3,
            # A CA-IMM forecast from a temporally-flow-matched, odometry-frame
            # motion track is admitted during its short bootstrap hold.  This
            # does not bypass the tracker or introduce a learned manoeuvre.
            "allow_temporal_flow_provisional": True,
            # Stable accumulated Livox frames exposed cluster-centre hopping
            # on static furniture as apparently coherent motion.  Admit a new
            # dynamic label only after the independent scan-flow channel has
            # observed finite closing TTC (up to 6 s); the bounded hold keeps
            # lateral crossings continuous through identity fragmentation.
            "dynamic_classification_temporal_corroboration_enabled": True,
            "dynamic_classification_temporal_corroboration_maximum_ttc_s": 6.0,
            "dynamic_classification_temporal_corroboration_hold_cycles": 12,
            "dynamic_classification_temporal_corroboration_minimum_support_beams": 3,
            # A strict, robot-relative collision-course certificate may
            # publish a coherent human track before radial scan flow becomes
            # observable.  This is the early lateral-crossing path; ordinary
            # compact/background motion still requires temporal corroboration.
            "dynamic_classification_collision_course_bypass_enabled": True,
            # Retain a strict collision-course forecast while the mapless
            # classifier is still accumulating its dynamic label.  This is a
            # perception continuity path, not a semantic control override.
            "allow_collision_course_provisional": True,
            # Publish a bounded person-compatible provisional forecast after
            # two consecutive, shape-backed motion frames.  This closes the
            # low-level-forecast -> mapless-label gap without admitting one-
            # frame unknown/static background motion.
            "person_provisional_enabled": bool(
                person_gate.get("enabled", True)
            ),
            "person_provisional_minimum_streak": int(
                person_gate.get("minimum_streak", 2)
            ),
            "person_provisional_minimum_samples": int(
                person_gate.get("minimum_samples", 4)
            ),
            "person_provisional_minimum_duration_s": float(
                person_gate.get("minimum_duration_s", 0.35)
            ),
            "person_provisional_minimum_speed_mps": float(
                person_gate.get("minimum_speed_mps", 0.18)
            ),
            "person_provisional_maximum_speed_mps": float(
                person_gate.get("maximum_speed_mps", 1.40)
            ),
            "person_provisional_minimum_displacement_m": float(
                person_gate.get("minimum_displacement_m", 0.12)
            ),
            "person_provisional_minimum_direction_coherence": float(
                person_gate.get("minimum_direction_coherence", 0.55)
            ),
            "person_provisional_maximum_fit_residual_m": float(
                person_gate.get("maximum_fit_residual_m", 0.08)
            ),
            "person_provisional_maximum_step_m": float(
                person_gate.get("maximum_step_m", 0.28)
            ),
            "person_provisional_minimum_support_beams": int(
                person_gate.get("minimum_support_beams", 3)
            ),
            "person_provisional_maximum_extent_m": float(
                person_gate.get("maximum_extent_m", 1.20)
            ),
            "human_point_cloud": point_cloud_gate,
        }
    bootstrap_overrides = {}
    if human_leg_mode:
        # Preserve the profile used by the immediately preceding physical run.
        # Active-passage validation must not silently change the predictor.
        bootstrap_overrides = {
            "required_motion_intervals": 2,
            "minimum_total_displacement_m": 0.06,
            "temporal_flow_threat_preemption_enabled": True,
            "temporal_flow_threat_maximum_ttc_s": 3.0,
            "temporal_flow_threat_minimum_support_beams": 3,
            "temporal_flow_threat_angle_tolerance_deg": 25.0,
            "temporal_flow_threat_range_tolerance_m": 0.75,
            "temporal_flow_threat_hold_cycles": 12,
            # Do not clear an initialized person track to make room for a
            # fragmented flow cluster; ordinary retirement still releases
            # genuinely stale slots.
            "temporal_flow_threat_preemption_allow_active_reset": False,
        }
    motion_bootstrap = MotionBootstrapMultiObstacleTracker.from_existing(
        base, **bootstrap_overrides
    )
    perception.dynamic_obstacle_tracker = MaplessStaticDynamicFilter.from_existing(
        motion_bootstrap, **overrides
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--goal-x", type=float, default=3.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--period-s", type=float, default=0.20)
    parser.add_argument("--accumulation-s", type=float, default=0.08)
    parser.add_argument("--livox-port", type=int, default=57701)
    parser.add_argument("--lidar-config", type=Path, required=True)
    parser.add_argument("--zero-can", action="store_true")
    parser.add_argument("--guarded-can", action="store_true")
    parser.add_argument("--max-v-mps", type=float, default=0.05)
    parser.add_argument("--max-omega-radps", type=float, default=0.10)
    parser.add_argument("--watchdog-timeout-s", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration_s <= 0.0 or args.period_s <= 0.0:
        raise ValueError("duration and period must be positive")
    if args.zero_can and args.guarded_can:
        raise ValueError("zero CAN and guarded CAN are mutually exclusive")
    args.output.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = build_pi5_full_config(
        PROJECT_ROOT, seed=args.seed, goal_x=args.goal_x, goal_y=args.goal_y
    )
    enable_single_final_control_authority(config)
    if args.guarded_can:
        config["real_robot_deployment"].update({
            "publish_enabled": True,
            "can_mode": "guarded_low_speed",
            "max_v_mps": float(args.max_v_mps),
            "max_omega_radps": float(args.max_omega_radps),
            "watchdog_timeout_s": float(args.watchdog_timeout_s),
        })
    # Keep the controller's frozen simulation timestep independent from the
    # slower wall-clock cadence of the Pi shadow loop.  Overwriting only the
    # MPPI timestep makes live tracker forecasts (forecast_dt_s) invalid as
    # soon as a dynamic track is first confirmed.
    controller_dt_s = float(config["experiment"]["control_dt"])
    forecast_dt_s = float(
        config["perception"]["dynamic_obstacle_tracker"]["forecast_dt_s"]
    )
    if abs(controller_dt_s - forecast_dt_s) > 1.0e-12:
        raise RuntimeError(
            "controller and dynamic forecast timesteps disagree: %.9g vs %.9g"
            % (controller_dt_s, forecast_dt_s)
        )
    config_path = args.output / "config_resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    adapter_config = _load_adapter_config(args.lidar_config)
    adapter = LivoxScanAdapter(adapter_config)
    components = make_components(config, PROJECT_ROOT)
    controller = components["controller"]
    perception = components["perception"]
    safety = components["safety"]
    reference = components["reference"]
    _install_mapless_tracker(perception)
    controller.reset(args.seed)
    perception.reset()
    safety.reset()

    timings = {name: [] for name in ("receive", "scan", "perception", "plan", "total")}
    rows = []
    deadline_misses = 0
    if args.zero_can:
        can_context = ScoutZeroOnlyCanGuard("can0")
    elif args.guarded_can:
        can_context = ScoutGuardedCanGateway(
            "can0",
            max_v_mps=args.max_v_mps,
            max_omega_radps=args.max_omega_radps,
            watchdog_timeout_s=args.watchdog_timeout_s,
        )
    else:
        can_context = nullcontext(None)
    started = time.monotonic()
    log_path = args.output / "cycles.jsonl"
    try:
        with can_context as can_guard, LivoxUdpReceiver(
            port=args.livox_port, timeout_s=max(0.25, args.period_s)
        ) as receiver, log_path.open("w", encoding="utf-8") as stream:
            if args.guarded_can:
                preflight_deadline = time.monotonic() + 2.0
                while time.monotonic() < preflight_deadline:
                    snapshot = can_guard.snapshot()
                    if snapshot.battery_v is not None and snapshot.fault is not None:
                        break
                    time.sleep(0.02)
                snapshot = can_guard.snapshot()
                if snapshot.battery_v is None or snapshot.fault is None:
                    raise RuntimeError("guarded CAN preflight received no chassis status")
                if snapshot.fault != 0:
                    raise RuntimeError(
                        "guarded CAN preflight requires fault=0, received %d"
                        % snapshot.fault
                    )
                if snapshot.battery_v < 22.0:
                    raise RuntimeError(
                        "guarded CAN preflight battery is too low: %.1f V"
                        % snapshot.battery_v
                    )
                can_guard.arm()
            next_cycle = time.monotonic()
            pose_x = 0.0
            pose_y = 0.0
            pose_yaw = 0.0
            last_pose_update = time.monotonic()
            while time.monotonic() - started < args.duration_s:
                cycle_start = time.perf_counter()
                stage = time.perf_counter()
                frame = receiver.receive_frame(args.accumulation_s)
                timings["receive"].append(1000.0 * (time.perf_counter() - stage))
                stage = time.perf_counter()
                scan, scan_diagnostics, human_points_base = (
                    adapter.convert_with_points(frame)
                )
                timings["scan"].append(1000.0 * (time.perf_counter() - stage))
                can_snapshot = None if can_guard is None else can_guard.snapshot()
                v_mps = 0.0 if can_snapshot is None else can_snapshot.v_mps
                omega = 0.0 if can_snapshot is None else can_snapshot.omega_radps
                pose_now = time.monotonic()
                pose_dt = max(0.0, min(1.0, pose_now - last_pose_update))
                if args.guarded_can:
                    pose_x += v_mps * math.cos(pose_yaw) * pose_dt
                    pose_y += v_mps * math.sin(pose_yaw) * pose_dt
                    pose_yaw += omega * pose_dt
                    pose_yaw = math.atan2(math.sin(pose_yaw), math.cos(pose_yaw))
                last_pose_update = pose_now
                observation = RobotObservation(
                    timestamp=float(scan.timestamp),
                    pose=Pose2D(pose_x, pose_y, pose_yaw),
                    twist=Twist2D(v_mps, omega),
                    scan=scan,
                    auxiliary={
                        "real_robot": True,
                        "shadow_mode": not args.guarded_can,
                        "simulator_truth_used": False,
                        "human_point_cloud_base": human_points_base,
                    },
                )
                stage = time.perf_counter()
                perceived = perception.process(observation)
                timings["perception"].append(1000.0 * (time.perf_counter() - stage))
                stage = time.perf_counter()
                plan = controller.plan(perceived.observation, reference)
                timings["plan"].append(1000.0 * (time.perf_counter() - stage))
                decision = safety.arbitrate(
                    plan.proposed_control, perceived.guard, plan.diagnostics
                )
                if args.guarded_can:
                    applied_v, applied_omega = can_guard.command(
                        decision.executed_control.v,
                        decision.executed_control.omega,
                    )
                else:
                    applied_v, applied_omega = 0.0, 0.0
                elapsed_ms = 1000.0 * (time.perf_counter() - cycle_start)
                timings["total"].append(elapsed_ms)
                if elapsed_ms > 1000.0 * args.period_s:
                    deadline_misses += 1
                tracker = dict(perceived.diagnostics.get(
                    "dynamic_obstacle_tracker", {}
                ))
                row = {
                    "cycle": len(rows),
                    "timestamp": observation.timestamp,
                    "raw_point_count": int(frame.points.shape[0]),
                    "scan": _json_value(scan_diagnostics.__dict__),
                    "local_obstacle_count": len(perceived.observation.local_obstacles),
                    "tracker_cluster_count": tracker.get("cluster_count"),
                    "tracker_track_count": tracker.get("track_count"),
                    "tracker_dynamic_indices": tracker.get(
                        "mapless_dynamic_track_indices", ()
                    ),
                    "tracker_static_indices": tracker.get(
                        "mapless_static_track_indices", ()
                    ),
                    "tracker_unknown_indices": tracker.get(
                        "mapless_unknown_track_indices", ()
                    ),
                    "forecast_count": tracker.get("valid_forecast_count", 0),
                    "proposed_v": plan.proposed_control.v,
                    "proposed_omega": plan.proposed_control.omega,
                    "arbitrated_v": decision.executed_control.v,
                    "arbitrated_omega": decision.executed_control.omega,
                    "applied_v": applied_v,
                    "applied_omega": applied_omega,
                    "safety_reason": decision.reason,
                    "can_published_motion": bool(args.guarded_can),
                    "timing_ms": {
                        name: values[-1] for name, values in timings.items()
                    },
                }
                rows.append(row)
                stream.write(json.dumps(_json_value(row), sort_keys=True) + "\n")
                stream.flush()
                print(
                    "cycle=%d points=%d clusters=%s dynamic=%s "
                    "forecast=%s cmd=(%.3f,%.3f) total=%.1fms"
                    % (
                        row["cycle"], row["raw_point_count"],
                        row["tracker_cluster_count"], row["tracker_dynamic_indices"],
                        row["forecast_count"], row["arbitrated_v"],
                        row["arbitrated_omega"], elapsed_ms,
                    ),
                    flush=True,
                )
                next_cycle += args.period_s
                delay = next_cycle - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_cycle = time.monotonic()
    finally:
        summary = {
            "contract": "pi5_scout_full_proposed_silent_v1",
            "completed_cycles": len(rows),
            "deadline_misses": deadline_misses,
            "deadline_miss_rate": (
                0.0 if not rows else deadline_misses / float(len(rows))
            ),
            "timing": {name: _timing(values) for name, values in timings.items()},
            "config_sha256": _sha256(config_path),
            "lidar_config_sha256": _sha256(args.lidar_config),
            "can_contract": (
                "zero_only"
                if args.zero_can
                else "guarded_low_speed"
                if args.guarded_can
                else "not_opened"
            ),
            "nonzero_can_commands_sent": int(
                getattr(can_context, "nonzero_commands_sent", 0)
            ),
            "watchdog_zero_events": int(
                getattr(can_context, "watchdog_zero_events", 0)
            ),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "controller_dt_s": controller_dt_s,
            "scheduler_period_s": float(args.period_s),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
