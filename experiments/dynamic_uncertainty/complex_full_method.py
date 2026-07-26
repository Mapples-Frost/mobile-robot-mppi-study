"""Build the frozen B11 Full arm for mixed static/dynamic MuJoCo maps.

The complex-map adapter intentionally owns no navigation algorithm. It replaces
the single-obstacle scene/task with a selected frozen map, then applies one
shared mixed-environment interface override.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from experiments.dynamic_uncertainty.run_single_dynamic_obstacle_paper_v4 import (
    _load_yaml,
    _mapping,
    configure_arm,
)
from mobile_robot_mppi.core.config import deep_merge, load_yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "configs/research/single_dynamic_obstacle_paper_v4.yaml"
)
DEFAULT_COMMON = (
    ROOT / "configs/research/complex_mixed_full_method_common.yaml"
)
MAPS = {
    "chapter1": (
        ROOT
        / "configs/research/"
        "mujoco_irregular_spiral_three_dynamic_v1.yaml"
    ),
    "chapter2": (
        ROOT
        / "configs/research/"
        "mujoco_complex_static_three_dynamic_v2.yaml"
    ),
    "chapter3": (
        ROOT
        / "configs/research/"
        "mujoco_scattered_clutter_three_loop_v1.yaml"
    ),
}


def _plain_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping: %s" % path)
    return value


def _paper_full(seed, protocol_path):
    protocol = _load_yaml(protocol_path)
    base = load_yaml(ROOT / protocol["base_config"])
    stage3 = _mapping(ROOT / protocol["stage3_protocol"])
    stage4 = _mapping(ROOT / protocol["stage4_protocol"])
    block = {
        "split": "id",
        "seed": int(seed),
        "model_block": int(seed) % 3,
    }
    return configure_arm(
        protocol,
        block,
        "B11_full_proposed",
        base,
        stage3,
        stage4,
    )


def build_complex_full_config(
    map_name,
    seed,
    *,
    protocol_path=DEFAULT_PROTOCOL,
    common_path=DEFAULT_COMMON,
):
    """Return one resolved, 600-rollout B11 complex-map configuration."""

    if str(map_name) not in MAPS:
        raise ValueError(
            "unknown complex map %r; expected one of %s"
            % (map_name, sorted(MAPS))
        )
    seed = int(seed)
    full = _paper_full(seed, Path(protocol_path).resolve())
    map_config = load_yaml(MAPS[str(map_name)])
    common = _plain_yaml(Path(common_path).resolve())

    # Replace experimental geometry and reference wholesale so no field from
    # the single-obstacle task leaks into the complex scene.
    full["scene"] = deepcopy(map_config["scene"])
    full["task"] = deepcopy(map_config["task"])
    full["task"]["terminate_on_boundary_violation"] = False
    full["experiment"].update({
        "name": "complex_full__%s__seed%d" % (map_name, seed),
        "seed": seed,
        "max_steps": int(map_config["experiment"]["max_steps"]),
        "initial_state": deepcopy(
            map_config["experiment"]["initial_state"]
        ),
        "terminate_on_collision": True,
        "complex_map": str(map_name),
        "complex_map_source": str(MAPS[str(map_name)].resolve()),
    })
    full = deep_merge(full, {
        "perception": common["perception"],
        "planner": common["planner"],
    })
    full["complex_gate_contract"] = deepcopy(common["gate_contract"])
    full["complex_method_contract"] = {
        "arm": "B11_full_proposed",
        "core_method_modified": False,
        "global_reference": "soft_static_astar",
        "dynamic_obstacles_in_astar": False,
        "shared_common_override": str(Path(common_path).resolve()),
    }
    full["scope_guards"].update({
        "paper_v4": False,
        "complex_mixed_development": True,
        "formal_experiment_started": False,
    })

    obstacles = tuple(full["scene"]["obstacles"])
    static_count = sum(
        not isinstance(item.get("motion"), dict)
        for item in obstacles
    )
    dynamic_count = len(obstacles) - static_count
    if static_count <= 0 or dynamic_count != 3:
        raise ValueError(
            "complex map must contain static geometry and exactly 3 "
            "moving obstacles"
        )
    if not full["perception"]["dynamic_obstacle_tracker"].get(
        "known_static_filter_enabled", False
    ):
        raise ValueError("complex Full requires the static-return filter")
    if int(
        full["perception"]["dynamic_obstacle_tracker"]["maximum_tracks"]
    ) != 3:
        raise ValueError("complex Full requires exactly three track slots")
    if full["planner"].get("path_boundary_enabled", False):
        raise ValueError("complex Full cannot hard-enforce the A* route")
    if full["planner"].get(
        "path_boundary_candidate_filter_enabled", False
    ):
        raise ValueError("complex Full cannot hard-filter the A* corridor")
    iterations = int(
        full["planner"]["paper_rl_driven"]["iterations"]
    )
    rollouts = int(full["planner"]["num_samples"]) * iterations
    if rollouts != 600:
        raise ValueError("complex Full changed the 600-rollout contract")
    return full


__all__ = [
    "DEFAULT_COMMON",
    "DEFAULT_PROTOCOL",
    "MAPS",
    "build_complex_full_config",
]
