import itertools
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import (
    audit_reference_path,
    audit_static_scene,
    point_clearance,
)
from mobile_robot_mppi.simulation.model_factory import build_diff_drive_mjcf
from mobile_robot_mppi.simulation.mujoco_plant import MujocoDiffDrivePlant


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "mujoco_irregular_spiral_three_dynamic_v1.yaml"
)


def _cross(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _point_segment_distance(point, start, end):
    delta = end - start
    fraction = float(
        np.clip(np.dot(point - start, delta) / np.dot(delta, delta), 0.0, 1.0)
    )
    return float(np.linalg.norm(point - (start + fraction * delta)))


def _segment_distance(a, b, c, d):
    first = b - a
    second = d - c
    denominator = _cross(first, second)
    if abs(denominator) > 1.0e-12:
        offset = c - a
        first_fraction = _cross(offset, second) / denominator
        second_fraction = _cross(offset, first) / denominator
        if (
            0.0 <= first_fraction <= 1.0
            and 0.0 <= second_fraction <= 1.0
        ):
            return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _split(config):
    static = [
        item
        for item in config["scene"]["obstacles"]
        if item.get("motion") is None
    ]
    dynamic = [
        item
        for item in config["scene"]["obstacles"]
        if item.get("motion") is not None
    ]
    return static, dynamic


def test_spiral_scene_has_nested_irregular_topology_and_core_goal():
    config = load_yaml(CONFIG)
    static, dynamic = _split(config)
    design = config["spiral_scene_design"]

    assert len(static) == design["static_obstacle_count"] == 42
    assert len(dynamic) == design["dynamic_obstacle_count"] == 3
    assert all(item["type"] == "segment" for item in static)
    assert all(len(item["parts"]) == 2 for item in dynamic)
    assert len(config["task"]["points"]) == 29
    assert design["goal_inside_core_ring"]
    assert config["task"]["position"] == pytest.approx((0.0, 0.0))
    assert config["planner"]["num_samples"] == 600


def test_frozen_spiral_route_is_clear_and_globally_reachable():
    config = load_yaml(CONFIG)
    static, _ = _split(config)
    contract = config["spiral_scene_feasibility_contract"]
    scene = {"name": config["scene"]["name"], "obstacles": static}
    points = config["task"]["points"]
    footprint = float(contract["robot_footprint_radius_m"])
    margin = float(contract["static_clearance_margin_m"])

    reference = audit_reference_path(
        scene,
        points,
        footprint,
        margin=margin,
        sample_spacing=0.01,
    )
    free_space = audit_static_scene(
        scene,
        points[0],
        points[-1],
        footprint,
        margin=margin,
        resolution=0.05,
    )

    assert reference["path_clear"]
    assert reference["minimum_clearance"] >= margin
    assert reference["reference_length"] == pytest.approx(
        config["spiral_scene_design"]["nominal_reference_length_m"],
        abs=1.0e-9,
    )
    assert free_space["start_free"]
    assert free_space["goal_free"]
    assert free_space["path_exists"]


def test_dynamic_carriers_cross_route_and_keep_static_clearance():
    config = load_yaml(CONFIG)
    static, dynamic = _split(config)
    contract = config["spiral_scene_feasibility_contract"]
    points = np.asarray(config["task"]["points"], dtype=np.float64)

    for obstacle in dynamic:
        motion = obstacle["motion"]
        start = np.asarray(motion["start"], dtype=np.float64)
        end = np.asarray(motion["end"], dtype=np.float64)
        route_distance = min(
            _segment_distance(start, end, route_start, route_end)
            for route_start, route_end in zip(points[:-1], points[1:])
        )
        assert route_distance <= float(
            contract["dynamic_route_intersection_tolerance_m"]
        )

        jitter = float(motion["endpoint_jitter_m"])
        for signs in itertools.product((-1.0, 1.0), repeat=4):
            jittered_start = start + jitter * np.asarray(signs[:2])
            jittered_end = end + jitter * np.asarray(signs[2:])
            minimum = min(
                point_clearance(
                    *(jittered_start + fraction * (
                        jittered_end - jittered_start
                    )),
                    static,
                    float(obstacle["radius"]),
                )
                for fraction in np.linspace(0.0, 1.0, 601)
            )
            assert minimum >= (
                float(contract["dynamic_path_static_clearance_floor_m"])
                - 1.0e-9
            )


def test_fast_randomized_carriers_retain_a_crossing_window():
    config = load_yaml(CONFIG)
    _, dynamic = _split(config)
    contract = config["spiral_scene_feasibility_contract"]
    width = float(contract["certified_crossing_width_m"])
    required = (
        width / float(contract["maximum_robot_speed_mps"])
        + float(contract["actuator_margin_s"])
    )

    assert required == pytest.approx(
        contract["minimum_required_free_gap_s"]
    )
    for obstacle in dynamic:
        motion = obstacle["motion"]
        nominal_span = float(np.linalg.norm(
            np.asarray(motion["end"]) - np.asarray(motion["start"])
        ))
        worst_span = nominal_span - (
            2.0 * math.sqrt(2.0) * float(motion["endpoint_jitter_m"])
        )
        worst_period = (
            float(motion["period_s"])
            * float(contract["worst_period_scale"])
        )
        free_gap = worst_period * (
            0.5 - width / (2.0 * worst_span)
        )
        assert worst_span > width
        assert free_gap >= required


def test_segment_walls_and_composite_carriers_compile_and_collide():
    config = load_yaml(CONFIG)
    static, dynamic = _split(config)
    xml = build_diff_drive_mjcf(config["plant"], config["scene"])
    model = mujoco.MjModel.from_xml_string(xml)
    plant = MujocoDiffDrivePlant(config["plant"], config["scene"])

    assert model.nmocap == 3
    assert len(static) == 42
    for offset, obstacle in enumerate(dynamic, start=len(static)):
        main_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_%d" % offset
        )
        assert main_id >= 0
        for part_index in range(2):
            part_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "obstacle_%d_part_%d" % (offset, part_index),
            )
            assert part_id >= 0
            assert part_id in plant._obstacle_geom_ids
