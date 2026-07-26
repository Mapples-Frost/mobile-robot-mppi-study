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
    / "mujoco_complex_static_three_dynamic_v2.yaml"
)


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


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
        t_value = _cross(offset, second) / denominator
        u_value = _cross(offset, first) / denominator
        if 0.0 <= t_value <= 1.0 and 0.0 <= u_value <= 1.0:
            return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _split_obstacles(config):
    obstacles = config["scene"]["obstacles"]
    static = [item for item in obstacles if item.get("motion") is None]
    dynamic = [item for item in obstacles if item.get("motion") is not None]
    return static, dynamic


def test_complex_scene_has_four_gate_static_maze_and_three_dynamic_obstacles():
    config = load_yaml(CONFIG)
    static, dynamic = _split_obstacles(config)
    design = config["complex_scene_design"]

    assert len(static) == design["static_obstacle_count"] == 39
    assert len(dynamic) == design["dynamic_obstacle_count"] == 3
    assert all(item["type"] == "box" for item in static + dynamic)
    assert all(len(item["parts"]) == 2 for item in dynamic)
    assert config["task"]["type"] == "polyline"
    assert len(config["task"]["points"]) == 17
    assert config["task"]["corridor_half_width"] >= 0.8
    assert config["action_space"]["upper"][0] == pytest.approx(0.6)
    assert config["planner"]["num_samples"] == 600
    assert design["controller_observation"] == "causal_laser_scan_only"
    assert not design["simulator_future_truth_used_by_controller"]
    assert not design["obstacle_identity_exposed_to_controller"]


def test_static_maze_and_frozen_reference_are_geometrically_feasible():
    config = load_yaml(CONFIG)
    static, _ = _split_obstacles(config)
    contract = config["complex_scene_feasibility_contract"]
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
        config["complex_scene_design"]["nominal_reference_length_m"],
        abs=1.0e-9,
    )
    assert free_space["start_free"]
    assert free_space["goal_free"]
    assert free_space["path_exists"]


def test_every_dynamic_program_crosses_route_and_stays_clear_of_static_geometry():
    config = load_yaml(CONFIG)
    static, dynamic = _split_obstacles(config)
    points = np.asarray(config["task"]["points"], dtype=np.float64)
    contract = config["complex_scene_feasibility_contract"]
    route_tolerance = float(
        contract["dynamic_route_intersection_tolerance_m"]
    )
    clearance_floor = float(
        contract["dynamic_path_static_clearance_floor_m"]
    )

    for obstacle in dynamic:
        motion = obstacle["motion"]
        start = np.asarray(motion["start"], dtype=np.float64)
        end = np.asarray(motion["end"], dtype=np.float64)
        route_distance = min(
            _segment_distance(start, end, segment_start, segment_end)
            for segment_start, segment_end in zip(points[:-1], points[1:])
        )
        assert route_distance <= route_tolerance

        jitter = float(motion["endpoint_jitter_m"])
        radius = float(obstacle["radius"])
        # Audit all endpoint-jitter box corners. Clearance along each segment
        # is sampled more finely than the collision geometry tolerance.
        for signs in itertools.product((-1.0, 1.0), repeat=4):
            jittered_start = start + jitter * np.asarray(signs[:2])
            jittered_end = end + jitter * np.asarray(signs[2:])
            values = [
                point_clearance(
                    *(jittered_start + fraction * (
                        jittered_end - jittered_start
                    )),
                    static,
                    radius,
                )
                for fraction in np.linspace(0.0, 1.0, 501)
            ]
            assert min(values) >= clearance_floor - 1.0e-9


def test_each_randomized_motion_retains_a_physically_feasible_crossing_gap():
    config = load_yaml(CONFIG)
    _, dynamic = _split_obstacles(config)
    contract = config["complex_scene_feasibility_contract"]
    crossing_width = float(contract["certified_crossing_width_m"])
    speed = float(contract["maximum_robot_speed_mps"])
    actuator_margin = float(contract["actuator_margin_s"])
    required = crossing_width / speed + actuator_margin
    jitter = float(contract["endpoint_jitter_m"])
    scale = float(contract["worst_period_scale"])

    assert required == pytest.approx(
        contract["minimum_required_free_gap_s"]
    )
    for obstacle in dynamic:
        motion = obstacle["motion"]
        nominal_span = float(np.linalg.norm(
            np.asarray(motion["end"], dtype=np.float64)
            - np.asarray(motion["start"], dtype=np.float64)
        ))
        worst_span = nominal_span - 2.0 * math.sqrt(2.0) * jitter
        worst_period = float(motion["period_s"]) * scale
        worst_free_gap = worst_period * (
            0.5 - crossing_width / (2.0 * worst_span)
        )
        assert worst_span > crossing_width
        assert worst_free_gap >= required


def test_mujoco_model_compiles_with_three_mocap_bodies_and_scene_colors():
    config = load_yaml(CONFIG)
    xml = build_diff_drive_mjcf(config["plant"], config["scene"])
    model = mujoco.MjModel.from_xml_string(xml)
    plant = MujocoDiffDrivePlant(config["plant"], config["scene"])
    static, dynamic = _split_obstacles(config)

    assert model.nmocap == 3
    assert len(static) == 39
    for offset, obstacle in enumerate(dynamic, start=len(static)):
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_%d" % offset
        )
        assert geom_id >= 0
        assert model.geom_rgba[geom_id] == pytest.approx(obstacle["rgba"])
        for part_index in range(2):
            part_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "obstacle_%d_part_%d" % (offset, part_index),
            )
            assert part_id >= 0
            assert part_id in plant._obstacle_geom_ids
