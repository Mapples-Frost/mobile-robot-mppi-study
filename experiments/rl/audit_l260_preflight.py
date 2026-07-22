"""Fail-closed preflight for the corrected L260 Tracking curriculum."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.train_rl_sampling_prior import _scene_configs
from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import (
    audit_reference_path,
    audit_static_scene,
)
from mobile_robot_mppi.rl.environment import DirectControlEnv


PROTOCOL = "L260"
EXPECTED_CHECKPOINT_SHA256 = (
    "fc9166f5c3010156a7c9fad4cb9d155ac222447506ff2ebdfe6c2205250a1547"
)
DEFAULT_CHECKPOINT = ROOT / (
    "results/research_platform/rl/"
    "l257_path_preview_bc_anchor_seed20262333_120k_v1/"
    "checkpoints/step_000030000.pt"
)
MANIFEST_PATH = ROOT / "configs/research/l260_curriculum/manifest.json"
SEED_CONFIGS = tuple(
    ROOT / f"configs/rl/l260_corrected_tracking_seed{seed}.yaml"
    for seed in (20262501, 20262502, 20262503)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value) -> bool:
    return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())


def _audit_configs(manifest, checkpoint: Path):
    if "l258" in checkpoint.as_posix().lower():
        raise ValueError("L258 checkpoints are forbidden as L260 initialization")
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"initialization checkpoint SHA mismatch: {checkpoint_hash}"
        )
    expected_train = [record["path"] for record in manifest["maps"] if record["role"] == "train"]
    expected_validation = [
        record["path"] for record in manifest["maps"] if record["role"] == "validation"
    ]
    resolved = []
    for index, path in enumerate(SEED_CONFIGS):
        config = load_yaml(path)
        text = json.dumps(config, sort_keys=True).lower()
        if "l258_tracking_curriculum" in text:
            raise ValueError(f"forbidden L258 reference in {path}")
        training = config["rl"]["training"]
        expected_seed = 20262501 + index
        expected_validation_seed = 20263501 + index
        assertions = {
            "seed": int(training["seed"]) == expected_seed,
            "validation_seed_base": int(training["validation_seed_base"]) == expected_validation_seed,
            "total_steps": int(training["total_steps"]) == 20000,
            "checkpoint_interval": int(training["checkpoint_interval"]) == 10000,
            "evaluation_interval": int(training["evaluation_interval"]) == 10000,
            "save_replay_buffer": bool(training["save_replay_buffer"]),
            "bc_anchor_disabled": not bool(training["bc_anchor"]["enabled"]),
            "cuda": str(training["device"]) == "cuda",
            "nominal_physics_only": training.get("physics_domain_config") is None,
            "train_maps": list(training["scene_configs"]) == expected_train,
            "validation_maps": list(training["validation_scene_configs"]) == expected_validation,
            "map_isolation": set(training["scene_configs"]).isdisjoint(
                training["validation_scene_configs"]
            ),
            "all_maps_before_10k": [
                int(np.argmax(phase["scene_weights"]))
                for phase in training["curriculum"]["phases"][:6]
            ] == list(range(6)),
            "all_maps_before_20k": [
                int(np.argmax(phase["scene_weights"]))
                for phase in training["curriculum"]["phases"][6:]
            ] == list(range(6)),
            "path_anchor_curriculum": bool(
                training["initial_state_curriculum"]["enabled"]
            ),
        }
        failed = sorted(name for name, passed in assertions.items() if not passed)
        if failed:
            raise ValueError(f"{path.name} contract failed: {failed}")
        resolved.append({
            "config": str(path),
            "seed": expected_seed,
            "validation_seed_base": expected_validation_seed,
            "output_dir": config["experiment"]["output_dir"],
            "checks": assertions,
        })
    if len({item["output_dir"] for item in resolved}) != len(resolved):
        raise ValueError("L260 seed output directories must be unique")
    return checkpoint_hash, resolved


def _audit_maps(manifest):
    records = []
    for record in manifest["maps"]:
        path = ROOT / record["path"]
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = load_yaml(path)
        for field in ("initial_state",):
            if config["experiment"][field] != raw["experiment"][field]:
                raise ValueError(f"inherited experiment.{field} in {path.name}")
        for field in ("field_size", "obstacles"):
            if config["scene"][field] != raw["scene"][field]:
                raise ValueError(f"inherited scene.{field} in {path.name}")
        if config["task"]["points"] != raw["task"]["points"]:
            raise ValueError(f"inherited task.points in {path.name}")
        radius = float(config["plant"]["robot"]["collision_radius"])
        margin = float(manifest["minimum_path_clearance"])
        path_audit = audit_reference_path(
            config["scene"], config["task"]["points"], radius,
            margin=margin, sample_spacing=0.01,
        )
        connectivity = audit_static_scene(
            config["scene"], config["experiment"]["initial_state"][:2],
            config["task"]["points"][-1], radius,
            margin=margin, resolution=0.05,
        )
        if not path_audit["path_clear"]:
            raise ValueError(f"footprint path blocked in {path.name}: {path_audit}")
        for key in ("start_free", "goal_free", "path_exists"):
            if not connectivity[key]:
                raise ValueError(f"{key} failed in {path.name}")
        records.append({
            "name": record["name"],
            "role": record["role"],
            "path": record["path"],
            "geometry_sha256": record["geometry_sha256"],
            "minimum_footprint_clearance": path_audit["minimum_clearance"],
            "reference_length": path_audit["reference_length"],
            "connectivity": {key: connectivity[key] for key in ("start_free", "goal_free", "path_exists")},
        })
    if len({record["geometry_sha256"] for record in records}) != 9:
        raise ValueError("L260 geometry fingerprints are not unique")
    return records


def _mujoco_smoke(manifest, steps_per_map: int):
    base = load_yaml(SEED_CONFIGS[0])
    results = []
    for index, record in enumerate(manifest["maps"]):
        config = _scene_configs(base, [record["path"]])[0]
        config["experiment"]["max_steps"] = max(steps_per_map + 1, 5)
        config["experiment"]["initial_state_noise"] = [0.0] * len(
            config["experiment"]["initial_state"]
        )
        config["rl"]["training"]["initial_state_curriculum"] = {"enabled": False}
        env = DirectControlEnv(config, ROOT, seed=20264601 + index)
        try:
            observation, reset_info = env.reset(seed=20264601 + index)
            if not _finite(observation):
                raise ValueError(f"non-finite reset observation: {record['name']}")
            last = None
            for _ in range(steps_per_map):
                transition = env.step(np.asarray((0.0, 0.0), dtype=np.float64))
                observation, reward, terminated, truncated, info = transition
                numeric = (
                    observation, reward, info["minimum_clearance"],
                    info["cross_track_error"], info["path_progress"],
                    info["proposed_control"], info["executed_control"],
                )
                if not all(_finite(value) for value in numeric):
                    raise ValueError(f"non-finite MuJoCo transition: {record['name']}")
                if info["collision"]:
                    raise ValueError(f"immediate MuJoCo collision: {record['name']}")
                last = info
                if terminated or truncated:
                    break
            results.append({
                "name": record["name"],
                "reset_scene": reset_info["scene"],
                "steps": int(env.steps),
                "collision": bool(last["collision"]),
                "minimum_clearance": float(last["minimum_clearance"]),
                "cross_track_error": float(last["cross_track_error"]),
                "path_progress": float(last["path_progress"]),
                "finite": True,
            })
        finally:
            env.close()
    return results


def _plot_maps(manifest, target: Path):
    """Write a dependency-free SVG contact sheet for human geometry review."""

    panel = 320
    padding = 34
    scale = (panel - 2 * padding) / 8.0
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{3 * panel}" height="{3 * panel}" viewBox="0 0 {3 * panel} {3 * panel}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for index, record in enumerate(manifest["maps"]):
        column = index % 3
        row = index // 3
        ox, oy = column * panel, row * panel
        config = load_yaml(ROOT / record["path"])
        width, height = config["scene"]["field_size"]

        def sx(x):
            return ox + padding + float(x) * scale

        def sy(y):
            return oy + panel - padding - float(y) * scale

        pieces.append(
            f'<rect x="{sx(0):.2f}" y="{sy(height):.2f}" width="{width * scale:.2f}" height="{height * scale:.2f}" fill="#fafafa" stroke="#555" stroke-width="1"/>'
        )
        for obstacle in config["scene"]["obstacles"]:
            if obstacle["type"] == "cylinder":
                x, y = obstacle["position"]
                pieces.append(
                    f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="{float(obstacle["radius"]) * scale:.2f}" fill="#ef6c00" fill-opacity="0.7"/>'
                )
            elif obstacle["type"] == "box" and not obstacle.get("yaw", 0.0):
                x, y = obstacle["position"]
                half_x, half_y = obstacle["size"]
                pieces.append(
                    f'<rect x="{sx(x - half_x):.2f}" y="{sy(y + half_y):.2f}" width="{2 * float(half_x) * scale:.2f}" height="{2 * float(half_y) * scale:.2f}" fill="#ef6c00" fill-opacity="0.7"/>'
                )
        points = config["task"]["points"]
        path = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        pieces.append(
            f'<polyline points="{path}" fill="none" stroke="#1769aa" stroke-width="2" stroke-linejoin="round"/>'
        )
        for x, y in points:
            pieces.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.2" fill="#1769aa"/>')
        pieces.append(f'<rect x="{sx(points[0][0]) - 4:.2f}" y="{sy(points[0][1]) - 4:.2f}" width="8" height="8" fill="#2e7d32"/>')
        pieces.append(f'<circle cx="{sx(points[-1][0]):.2f}" cy="{sy(points[-1][1]):.2f}" r="5" fill="#c62828"/>')
        title = html.escape(f"{record['role']}: {record['family']}")
        pieces.append(f'<text x="{ox + panel / 2:.2f}" y="{oy + 21}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#222">{title}</text>')
    target.parent.mkdir(parents=True, exist_ok=True)
    pieces.append("</svg>")
    target.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/research_platform/rl/l260_preflight")
    parser.add_argument("--mujoco-smoke", action="store_true")
    parser.add_argument("--steps-per-map", type=int, default=3)
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checkpoint_hash, config_records = _audit_configs(manifest, args.checkpoint.resolve())
    map_records = _audit_maps(manifest)
    smoke = _mujoco_smoke(manifest, args.steps_per_map) if args.mujoco_smoke else []
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "l260_map_preflight.svg"
    _plot_maps(manifest, plot_path)
    payload = {
        "protocol": PROTOCOL,
        "status": "pass",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "config_checks": config_records,
        "map_checks": map_records,
        "mujoco_smoke": smoke,
        "plot": str(plot_path),
    }
    (output_dir / "preflight.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
