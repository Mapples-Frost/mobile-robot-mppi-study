import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "contrib"
    / "student_scout_mini"
    / "dynamic_mppi_sim_v9.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("student_dynamic_mppi_v9", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_map_cylinders_use_grid_center_and_keep_static_marker():
    module = _load_module()
    grid = np.zeros((80, 100), dtype=np.uint8)
    grid[37:43, 47:53] = 1
    cylinders = module.map_to_scene_cylinders(
        grid,
        {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
        min_cl=30,
        max_cl=100,
        margin=0.10,
    )

    assert len(cylinders) == 1
    x, y, radius, known_static = cylinders[0]
    assert abs(x) < 0.10
    assert abs(y) < 0.10
    assert radius > 0.10
    assert known_static is True
