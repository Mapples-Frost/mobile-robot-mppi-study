from pathlib import Path

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import audit_static_scene


ROOT = Path(__file__).resolve().parents[2]
SCENES = (
    "mujoco_strong_mppi_baseline.yaml",
    "mujoco_lab_complex.yaml",
    "mujoco_narrow_corridor.yaml",
    "mujoco_u_trap_long_board.yaml",
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
