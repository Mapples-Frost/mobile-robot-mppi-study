"""Fabrication-layout checks for the modular 6.5 m real-robot field."""

from __future__ import annotations

from collections import Counter, deque
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "real_robot" / "obstacle_kit_6p5m.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _point_clear(config, scene, x, y, extra_clearance=0.05):
    radius = float(config["design_assumptions"]["robot_collision_radius"]) + extra_clearance
    for item in scene.get("obstacles", []):
        module = config["modules"][item["module"]]
        cx, cy = item["center"]
        if module["kind"] == "wall":
            length, thickness, _ = module["dimensions"]
            yaw = math.radians(float(item.get("yaw_deg", 0.0)))
            dx, dy = x - cx, y - cy
            local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
            local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
            outside_x = max(abs(local_x) - length / 2.0, 0.0)
            outside_y = max(abs(local_y) - thickness / 2.0, 0.0)
            if math.hypot(outside_x, outside_y) <= radius:
                return False
        elif module["kind"] in ("octagonal_prism", "dynamic_octagonal_shell"):
            # The fabrication octagon is inscribed in this conservative outer
            # circle, so the reachability audit never overstates free space.
            if math.hypot(x - cx, y - cy) <= radius + float(module["diameter"]) / 2.0:
                return False
    return True


def _has_grid_path(config, scene, resolution=0.05):
    xmin, xmax, ymin, ymax = map(float, config["field"]["active_bounds"])
    radius = float(config["design_assumptions"]["robot_collision_radius"])
    xmin += radius; xmax -= radius; ymin += radius; ymax -= radius
    nx = int(round((xmax - xmin) / resolution)) + 1
    ny = int(round((ymax - ymin) / resolution)) + 1

    def index(point):
        return (int(round((point[0] - xmin) / resolution)), int(round((point[1] - ymin) / resolution)))

    def point(cell):
        return (xmin + cell[0] * resolution, ymin + cell[1] * resolution)

    start = index(config["field"]["start"])
    goal = index(config["field"]["goal"])
    queue = deque([start])
    visited = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in visited or not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny):
                continue
            px, py = point(nxt)
            if _point_clear(config, scene, px, py):
                visited.add(nxt)
                queue.append(nxt)
    return False


def test_every_static_layout_has_a_buffered_start_to_goal_path():
    config = _config()
    for name in (
        "clean_dynamics",
        "single_obstacle",
        "straight_corridor_0900",
        "u_trap",
        "lab_complex",
        "dynamic_crossing",
        "friction_patch",
    ):
        assert _has_grid_path(config, config["scenes"][name]), name


def test_no_scene_exceeds_procured_module_quantities():
    config = _config()
    available = {key: int(value["quantity"]) for key, value in config["modules"].items()}
    for name, scene in config["scenes"].items():
        used = Counter(item["module"] for item in scene.get("obstacles", []))
        if scene.get("dynamic_obstacle"):
            used[scene["dynamic_obstacle"]["module"]] += 1
        if scene.get("floor_patch"):
            used[scene["floor_patch"]["module"]] += 1
        for module, count in used.items():
            assert count <= available[module], (name, module, count, available[module])


def test_minimum_corridor_width_and_dynamic_speed_are_safe_by_design():
    config = _config()
    assumptions = config["design_assumptions"]
    corridor = config["scenes"]["straight_corridor_0900"]
    assert corridor["nominal_free_width"] >= assumptions["minimum_free_passage"]
    dynamic = config["scenes"]["dynamic_crossing"]["dynamic_obstacle"]
    assert max(dynamic["test_speeds"]) <= assumptions["maximum_dynamic_obstacle_speed"]


def test_field_transform_preserves_existing_goal_3_3_contract():
    config = _config()
    origin = config["field"]["experiment_origin_in_field"]
    goal = config["field"]["goal"]
    assert [goal[index] - origin[index] for index in range(2)] == [3.0, 3.0]


def test_acrylic_geometry_preserves_plan_footprints_and_lidar_coverage():
    config = _config()
    assumptions = config["design_assumptions"]
    obstacle_height = float(assumptions["obstacle_height"])
    margin = float(assumptions["minimum_lidar_vertical_margin"])
    beam_min, beam_max = map(float, assumptions["compatible_lidar_beam_height_range"])

    assert obstacle_height == 0.35
    assert beam_min >= margin
    assert beam_max + margin <= obstacle_height
    assert config["materials"]["wall_and_base"] == "6_mm_opaque_matte_cast_pmma"
    assert "transparent_pmma" in config["materials"]["prohibited"]

    for module_id in ("W600", "W300"):
        module = config["modules"][module_id]
        _, plan_depth, height = map(float, module["dimensions"])
        panel_length, panel_height, panel_thickness = map(float, module["panel_dimensions"])
        base_length, fabricated_depth, base_thickness = map(float, module["base_dimensions"])
        assert plan_depth == float(module["base_depth"]) == 0.16
        assert height == obstacle_height
        assert float(module["panel_thickness"]) == 0.006
        assert panel_height + base_thickness == height
        assert panel_length == base_length == float(module["dimensions"][0])
        assert panel_thickness == base_thickness == 0.006
        assert fabricated_depth == plan_depth
        assert int(module["gusset_count"]) == 2 * int(module["gusset_pair_count"])
        assert len(module["gusset_station_x"]) == int(module["gusset_pair_count"])
        assert float(module["gusset_horizontal_leg"]) + panel_thickness / 2.0 <= plan_depth / 2.0 - 0.005

    for module_id in ("C240", "C320", "C560"):
        module = config["modules"][module_id]
        assert module["kind"] == "octagonal_prism"
        assert int(module["sides"]) == 8
        assert float(module["height"]) == obstacle_height
