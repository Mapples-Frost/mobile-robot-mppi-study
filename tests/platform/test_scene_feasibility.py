from pathlib import Path

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import (
    audit_reference_path,
    audit_static_scene,
)


ROOT = Path(__file__).resolve().parents[2]
SCENES = (
    "mujoco_strong_mppi_baseline.yaml",
    "mujoco_lab_complex.yaml",
    "mujoco_narrow_corridor.yaml",
    "mujoco_u_trap_long_board.yaml",
)

L218_SCENES = (
    "mujoco_l218_serpentine_polyline.yaml",
    "mujoco_l218_giant_u_polyline.yaml",
    "mujoco_l218_opposed_u_polyline.yaml",
    "mujoco_l218_nested_u_polyline.yaml",
    "mujoco_l218_cylinder_forest_polyline.yaml",
    "mujoco_l218_cylinder_spiral_polyline.yaml",
)


def test_research_static_scenes_have_free_start_goal_and_a_geometric_path():
    for filename in SCENES:
        config = load_yaml(ROOT / "configs/research" / filename)
        report = audit_static_scene(
            config["scene"],
            config["experiment"]["initial_state"][:2],
            config["task"]["position"],
            config["plant"]["robot"]["collision_radius"],
        )
        assert report["start_free"], filename
        assert report["goal_free"], filename
        assert report["path_exists"], filename


def test_l218_routes_are_geometrically_feasible_and_clear():
    for filename in L218_SCENES:
        config = load_yaml(ROOT / "configs/research" / filename)
        start = config["experiment"]["initial_state"][:2]
        goal = config["task"]["points"][-1]
        radius = config["plant"]["robot"]["collision_radius"]
        connectivity = audit_static_scene(
            config["scene"], start, goal, radius, margin=0.03
        )
        reference = audit_reference_path(
            config["scene"],
            config["task"]["points"],
            radius,
            margin=0.03,
            sample_spacing=0.01,
        )
        assert connectivity["start_free"], filename
        assert connectivity["goal_free"], filename
        assert connectivity["path_exists"], filename
        assert reference["path_clear"], (
            filename,
            reference["minimum_clearance"],
            reference["minimum_clearance_location"],
        )
