"""Blocked runtime screen for the qualified task-aware residual rollouts."""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import yaml

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.core.spaces import (
    action_spec_from_config,
    state_spec_from_config,
)
from mobile_robot_mppi.learning.models import (
    PlatformResidualDynamics,
    ResidualComponentMaskedDynamics,
)
from mobile_robot_mppi.planning.dynamics import (
    DynamicUnicyclePrediction,
    ResidualPrediction,
)
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT
    / "configs"
    / "research"
    / "dynamic_uncertainty_residual_runtime_stage3.yaml"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path, rows):
    if not rows:
        raise ValueError("runtime screen produced no rows")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_protocol(path):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("runtime protocol must be a mapping")
    return values


def _build_controller(base_config, block, protocol, arm):
    checkpoint = ROOT / str(block["checkpoint"])
    if _sha256(checkpoint) != str(block["sha256"]):
        raise ValueError("checkpoint hash mismatch: %s" % checkpoint)
    device = str(arm["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA arm requested but CUDA is unavailable")
    torch.set_num_threads(int(arm["torch_num_threads"]))
    residual = PlatformResidualDynamics.from_checkpoint(
        checkpoint,
        device=device,
        use_torchscript=bool(protocol["torchscript"]),
        device_rollout_enabled=bool(
            arm.get("residual_device_rollout_enabled", False)
        ),
        cuda_graph_enabled=bool(
            arm.get("residual_cuda_graph_enabled", False)
        ),
    )
    residual = ResidualComponentMaskedDynamics(
        residual, protocol["residual_component_mask"]
    )
    plant = base_config["plant"]
    nominal = DynamicUnicyclePrediction(
        plant.get("nominal_velocity_time_constant", 0.18),
        plant.get("nominal_yaw_time_constant", 0.12),
    )
    dynamics = ResidualPrediction(nominal, residual)
    state_spec = state_spec_from_config(base_config["state_space"])
    action_spec = action_spec_from_config(base_config["action_space"])
    planner = dict(base_config["planner"])
    planner.setdefault("dt", base_config["experiment"]["control_dt"])
    planner.setdefault("seed", base_config["experiment"].get("seed", 0))
    config = MppiConfig.from_mapping(planner, action_spec.dimension)
    controller = MppiController(
        dynamics=dynamics,
        state_spec=state_spec,
        action_spec=action_spec,
        config=config,
    )
    return controller


def _control_batch(base_config, seed):
    state_spec = state_spec_from_config(base_config["state_space"])
    action_spec = action_spec_from_config(base_config["action_space"])
    planner = dict(base_config["planner"])
    horizon = int(planner["horizon"])
    count = int(planner["num_samples"])
    rng = np.random.RandomState(int(seed))
    controls = rng.uniform(
        action_spec.lower,
        action_spec.upper,
        size=(count, horizon, action_spec.dimension),
    )
    initial_state = np.zeros(state_spec.dimension, dtype=np.float64)
    if "v" in state_spec.names:
        initial_state[state_spec.index("v")] = 0.2
    return initial_state, controls


def _synchronize(device):
    if str(device) == "cuda":
        torch.cuda.synchronize()


def _benchmark_cell(controller, initial_state, controls, arm, warmups, repeats):
    device = str(arm["device"])
    trajectory = None
    for _ in range(int(warmups)):
        trajectory = controller.rollout(initial_state, controls)
    _synchronize(device)
    timings = []
    for _ in range(int(repeats)):
        _synchronize(device)
        started = time.perf_counter()
        trajectory = controller.rollout(initial_state, controls)
        _synchronize(device)
        timings.append(1000.0 * (time.perf_counter() - started))
    values = np.asarray(trajectory, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("runtime arm produced nonfinite trajectories")
    return values, timings


def _schedule(protocol):
    cells = [
        (block_index, arm_index)
        for block_index in range(len(protocol["residual_model_blocks"]))
        for arm_index in range(len(protocol["design"]["arms"]))
    ]
    rng = np.random.RandomState(int(protocol["design"]["schedule_seed"]))
    return [cells[index] for index in rng.permutation(len(cells))]


def _summarize(rows, trajectories, protocol):
    baseline_name = str(
        protocol["screen_gate"].get("baseline_arm", "cpu_1_thread")
    )
    tolerance = float(
        protocol["screen_gate"]["maximum_trajectory_absolute_difference"]
    )
    for row in rows:
        key = (int(row["model_block"]), str(row["arm"]))
        baseline = trajectories[(int(row["model_block"]), baseline_name)]
        difference = np.abs(trajectories[key] - baseline)
        row["maximum_trajectory_absolute_difference"] = float(
            np.max(difference)
        )
        row["mean_trajectory_absolute_difference"] = float(
            np.mean(difference)
        )
        row["trajectory_equivalent"] = bool(
            row["maximum_trajectory_absolute_difference"] <= tolerance
        )

    baseline_by_block = {
        int(row["model_block"]): float(row["p95_ms"])
        for row in rows
        if row["arm"] == baseline_name
    }
    arm_summary = []
    for arm in protocol["design"]["arms"]:
        name = str(arm["name"])
        selected = [row for row in rows if row["arm"] == name]
        worst_p95 = max(float(row["p95_ms"]) for row in selected)
        worst_baseline = max(baseline_by_block.values())
        speedup = 1.0 - worst_p95 / worst_baseline
        equivalent = all(bool(row["trajectory_equivalent"]) for row in selected)
        arm_summary.append(
            {
                "arm": name,
                "device": str(arm["device"]),
                "torch_num_threads": int(arm["torch_num_threads"]),
                "residual_device_rollout_enabled": bool(
                    arm.get("residual_device_rollout_enabled", False)
                ),
                "residual_cuda_graph_enabled": bool(
                    arm.get("residual_cuda_graph_enabled", False)
                ),
                "worst_block_p95_ms": worst_p95,
                "worst_block_p95_speedup_fraction": float(speedup),
                "trajectory_equivalent_on_all_blocks": equivalent,
                "finite_output_on_all_blocks": True,
                "eligible": bool(
                    equivalent
                    and speedup
                    >= float(
                        protocol["screen_gate"][
                            "minimum_worst_block_p95_speedup_fraction"
                        ]
                    )
                    and worst_p95
                    <= float(
                        protocol["screen_gate"].get(
                            "maximum_selected_worst_block_p95_ms",
                            float("inf"),
                        )
                    )
                ),
            }
        )
    eligible = [row for row in arm_summary if row["eligible"]]
    selected = (
        min(eligible, key=lambda row: row["worst_block_p95_ms"])
        if eligible
        else None
    )
    return arm_summary, selected


def run(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol = _load_protocol(protocol_path)
    base_config = load_yaml(ROOT / str(protocol["base_config"]))
    output_dir = (ROOT / str(protocol["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_state, controls = _control_batch(
        base_config, protocol["design"]["common_control_seed"]
    )
    rows = []
    trajectories = {}
    schedule = _schedule(protocol)
    for run_order, (block_index, arm_index) in enumerate(schedule):
        block = protocol["residual_model_blocks"][block_index]
        arm = protocol["design"]["arms"][arm_index]
        print(
            "[%d/%d] block=%d arm=%s"
            % (run_order + 1, len(schedule), block_index, arm["name"]),
            flush=True,
        )
        controller = _build_controller(base_config, block, protocol, arm)
        trajectory, timings = _benchmark_cell(
            controller,
            initial_state,
            controls,
            arm,
            protocol["design"]["warmup_rollouts"],
            protocol["design"]["measured_rollouts"],
        )
        trajectories[(block_index, str(arm["name"]))] = trajectory
        rows.append(
            {
                "run_order": int(run_order),
                "model_block": int(block_index),
                "checkpoint_seed": int(block["seed"]),
                "arm": str(arm["name"]),
                "device": str(arm["device"]),
                "torch_num_threads": int(arm["torch_num_threads"]),
                "residual_device_rollout_enabled": bool(
                    arm.get("residual_device_rollout_enabled", False)
                ),
                "residual_cuda_graph_enabled": bool(
                    arm.get("residual_cuda_graph_enabled", False)
                ),
                "repeat_count": len(timings),
                "mean_ms": float(np.mean(timings)),
                "median_ms": float(np.median(timings)),
                "p95_ms": _percentile(timings, 95),
                "minimum_ms": float(np.min(timings)),
                "maximum_ms": float(np.max(timings)),
            }
        )
        del controller
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    arm_summary, selected = _summarize(rows, trajectories, protocol)
    result = {
        "schema_version": 1,
        "stage": "residual_runtime_stage3_screen",
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": _sha256(protocol_path),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "schedule": [
            {
                "run_order": index,
                "model_block": int(block),
                "arm": str(protocol["design"]["arms"][arm]["name"]),
            }
            for index, (block, arm) in enumerate(schedule)
        ],
        "rows": rows,
        "arm_summary": arm_summary,
        "selected_arm": selected,
        "sealed_seeds_opened": False,
        "rl_enabled": False,
    }
    _write_json(output_dir / "runtime_screen.json", result)
    _write_csv(output_dir / "runtime_cells.csv", rows)
    _write_csv(output_dir / "runtime_arm_summary.csv", arm_summary)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    result = run(args.protocol)
    print(json.dumps(result["selected_arm"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
