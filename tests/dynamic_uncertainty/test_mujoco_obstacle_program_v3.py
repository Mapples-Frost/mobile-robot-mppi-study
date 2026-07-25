import importlib.util
from pathlib import Path

import mujoco
import yaml

from mobile_robot_mppi.obstacles.motion import noise_profiles_from_mapping
from mobile_robot_mppi.obstacles.patrol import generate_patrol_trajectory


ROOT = Path(__file__).resolve().parents[2]
VIEWER_PATH = (
    ROOT
    / "experiments/dynamic_uncertainty/"
    "show_mujoco_obstacle_program_v3.py"
)
CONFIG_PATH = ROOT / "configs/research/dynamic_obstacle_process_v3.yaml"


def _load_viewer_module():
    spec = importlib.util.spec_from_file_location("v3_viewer", VIEWER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_multiepisode_mujoco_model_compiles_and_updates():
    viewer = _load_viewer_module()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    profiles = noise_profiles_from_mapping(config["noise_profiles"])
    trajectory = generate_patrol_trajectory(
        "hybrid_patrol", 730100021, profiles["medium"], config
    )
    model = mujoco.MjModel.from_xml_string(viewer.build_multiepisode_xml())
    data = mujoco.MjData(model)
    assert model.nmocap == 1
    viewer.configure_episode_scene(model, trajectory, 2, 4)
    scene = mujoco.MjvScene(model, maxgeom=4)
    viewer.configure_episode_label(
        scene, trajectory, "hybrid_patrol", 2, 4, 730100021
    )
    assert scene.ngeom == 1
    assert "Episode 3/4" in scene.geoms[0].label
    assert "hybrid_patrol" in scene.geoms[0].label
    viewer._set_obstacle_pose(data, trajectory.states[0])
    mujoco.mj_forward(model, data)
