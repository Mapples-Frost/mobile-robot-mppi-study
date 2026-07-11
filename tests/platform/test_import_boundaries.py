import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_package_root_is_torch_mujoco_and_ros_free():
    code = (
        "import sys; import mobile_robot_mppi; "
        "assert 'torch' not in sys.modules; "
        "assert 'mujoco' not in sys.modules; "
        "assert 'rospy' not in sys.modules"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.check_call([sys.executable, "-c", code], cwd=str(ROOT), env=environment)


def test_core_package_does_not_depend_on_experiments():
    code = (
        "import sys; from mobile_robot_mppi.core import PointGoal, StateSpec; "
        "assert not any(name == 'experiments' or name.startswith('experiments.') for name in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.check_call([sys.executable, "-c", code], cwd=str(ROOT), env=environment)
