import itertools
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
from mobile_robot_mppi.perception.legacy_pipeline import LegacyScanPipeline
from mobile_robot_mppi.runtime.factories import make_components


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "mujoco_scattered_clutter_three_loop_v1.yaml"
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


def _polygon_area(points):
    values = np.asarray(points, dtype=np.float64)
    return 0.5 * abs(
        float(
            np.dot(values[:, 0], np.roll(values[:, 1], -1))
            - np.dot(values[:, 1], np.roll(values[:, 0], -1))
        )
    )


def test_chapter3_is_hidden_astar_clutter_without_visible_route():
    config = load_yaml(CONFIG)
    static, dynamic = _split(config)
    design = config["chapter3_scene_design"]

    assert len(static) == design["static_obstacle_count"] == 34
    assert len(dynamic) == design["dynamic_obstacle_count"] == 3
    assert config["task"]["type"] == "polyline"
    assert config["experiment"]["initial_state"][:2] == [-6.0, -4.0]
    assert config["task"]["position"] == [6.0, 4.0]
    assert not config["scene"]["visual"]["show_reference_overlay"]
    assert design["controller_reference"] == "hidden_static_astar_polyline"
    assert not design["obvious_corridor_walls"]
    assert config["planner"]["num_samples"] == 600
    assert config["planner"]["known_static_map_cost_enabled"]


def test_static_mask_rejects_map_surface_but_not_nearby_dynamic_return():
    config = load_yaml(CONFIG)
    static, _ = _split(config)
    perception_config = dict(config["perception"])
    tracker_config = dict(
        perception_config["dynamic_obstacle_tracker"]
    )
    tracker_config["known_static_obstacles"] = static
    perception_config["dynamic_obstacle_tracker"] = tracker_config
    pipeline = LegacyScanPipeline(ROOT, perception_config)

    box = next(item for item in static if item["type"] == "box")
    center = np.asarray(box["position"], dtype=np.float64)
    yaw = float(box["yaw"])
    outward = np.asarray((np.cos(yaw), np.sin(yaw)))
    surface = center + float(box["size"][0]) * outward
    nearby_dynamic = surface + 0.04 * outward
    mask = pipeline._known_static_hit_mask(
        np.stack((surface, nearby_dynamic))
    )

    assert mask.tolist() == [True, False]


def test_exact_static_map_cost_uses_signed_footprint_clearance():
    config = load_yaml(CONFIG)
    static, _ = _split(config)
    components = make_components(config, ROOT)
    controller = components["controller"]
    box = next(item for item in static if item["type"] == "box")
    center = np.asarray(box["position"], dtype=np.float64)
    yaw = float(box["yaw"])
    outward = np.asarray((np.cos(yaw), np.sin(yaw)))
    surface = center + float(box["size"][0]) * outward
    trajectories = np.zeros(
        (2, controller.config.horizon + 1, 5), dtype=np.float64
    )
    trajectories[0, :, :2] = center
    trajectories[1, :, :2] = (
        surface
        + (controller.config.robot_radius + 0.02) * outward
    )
    try:
        clearance = controller._known_static_map_clearance(
            trajectories, static
        )
    finally:
        close_controller = getattr(controller, "close", None)
        if callable(close_controller):
            close_controller()
        components["plant"].close()

    assert np.max(clearance[0]) < 0.0
    assert np.min(clearance[1]) == pytest.approx(0.02)


def test_scattered_scene_is_reachable_but_straight_line_is_blocked():
    config = load_yaml(CONFIG)
    static, _ = _split(config)
    contract = config["chapter3_feasibility_contract"]
    scene = {"name": config["scene"]["name"], "obstacles": static}
    start = config["experiment"]["initial_state"][:2]
    goal = config["task"]["position"]
    footprint = float(contract["robot_footprint_radius_m"])
    margin = float(contract["static_clearance_margin_m"])

    free_space = audit_static_scene(
        scene,
        start,
        goal,
        footprint,
        margin=margin,
        resolution=float(contract["occupancy_resolution_m"]),
    )
    direct = audit_reference_path(
        scene,
        [start, goal],
        footprint,
        margin=margin,
        sample_spacing=0.01,
    )

    assert free_space["start_free"]
    assert free_space["goal_free"]
    assert free_space["path_exists"]
    assert not direct["path_clear"]


def test_closed_loops_are_non_degenerate_and_not_ping_pong_paths():
    config = load_yaml(CONFIG)
    _, dynamic = _split(config)
    design = config["chapter3_scene_design"]
    contract = config["chapter3_feasibility_contract"]

    assert [
        len(item["motion"]["waypoints"]) for item in dynamic
    ] == design["closed_loop_waypoint_counts"]
    for obstacle in dynamic:
        motion = obstacle["motion"]
        points = np.asarray(motion["waypoints"], dtype=np.float64)
        deltas = np.roll(points, -1, axis=0) - points
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        heading_changes = np.angle(
            np.exp(1.0j * (np.roll(headings, -1) - headings))
        )

        assert motion["type"] == "closed_waypoint_loop"
        assert _polygon_area(points) >= float(
            contract["minimum_dynamic_loop_area_m2"]
        )
        assert np.sum(np.abs(heading_changes) > 0.20) >= int(
            contract["minimum_dynamic_heading_changes"]
        )
        assert len(set(motion["segment_durations_s"])) > 1


def test_loop_jitter_envelopes_stay_clear_of_static_clutter():
    config = load_yaml(CONFIG)
    static, dynamic = _split(config)
    floor = float(
        config["chapter3_feasibility_contract"][
            "dynamic_path_static_clearance_floor_m"
        ]
    )

    for obstacle in dynamic:
        motion = obstacle["motion"]
        points = np.asarray(motion["waypoints"], dtype=np.float64)
        jitter = float(motion["waypoint_jitter_m"])
        for start, end in zip(points, np.roll(points, -1, axis=0)):
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
                    for fraction in np.linspace(0.0, 1.0, 401)
                )
                assert minimum >= floor - 1.0e-9


def test_closed_loop_runtime_randomizes_reproducibly_and_turns():
    config = load_yaml(CONFIG)
    initial = np.asarray(
        config["experiment"]["initial_state"], dtype=np.float64
    )
    first = MujocoDiffDrivePlant(config["plant"], config["scene"])
    second = MujocoDiffDrivePlant(config["plant"], config["scene"])
    first.reset(783100001, initial)
    second.reset(783100001, initial)

    for left, right in zip(
        first._episode_dynamic_obstacles,
        second._episode_dynamic_obstacles,
    ):
        assert left["motion_type"] == "closed_waypoint_loop"
        assert left["waypoints"] == pytest.approx(right["waypoints"])
        assert left["segment_durations_s"] == pytest.approx(
            right["segment_durations_s"]
        )
        assert left["phase_s"] == pytest.approx(right["phase_s"])

    positions = []
    yaws = []
    item = first._episode_dynamic_obstacles[1]
    for timestamp in np.linspace(0.0, item["period_s"], 13)[:-1]:
        first._set_dynamic_obstacles(float(timestamp))
        positions.append(first.data.mocap_pos[item["mocap_id"], :2].copy())
        quaternion = first.data.mocap_quat[item["mocap_id"]]
        yaws.append(
            2.0 * np.arctan2(float(quaternion[3]), float(quaternion[0]))
        )

    assert np.ptp(np.asarray(positions)[:, 0]) > 1.5
    assert np.ptp(np.asarray(positions)[:, 1]) > 1.5
    assert len(np.unique(np.round(np.unwrap(yaws), 2))) >= 4


def test_chapter3_mujoco_model_has_three_composite_mocap_bodies():
    config = load_yaml(CONFIG)
    static, dynamic = _split(config)
    xml = build_diff_drive_mjcf(config["plant"], config["scene"])
    model = mujoco.MjModel.from_xml_string(xml)
    plant = MujocoDiffDrivePlant(config["plant"], config["scene"])

    assert model.nmocap == 3
    for offset, _ in enumerate(dynamic, start=len(static)):
        for part_index in range(2):
            part_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "obstacle_%d_part_%d" % (offset, part_index),
            )
            assert part_id >= 0
            assert part_id in plant._obstacle_geom_ids
