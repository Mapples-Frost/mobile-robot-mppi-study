from pathlib import Path

import pytest

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import audit_reference_path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = (
    "mujoco_l222_serpentine_safe_polyline.yaml",
    "mujoco_l222_nested_u_safe_polyline.yaml",
    "mujoco_l222_cylinder_spiral_safe_polyline.yaml",
)


@pytest.mark.parametrize("filename", CONFIGS)
def test_l222_reference_respects_unchanged_scan_guard_envelope(filename):
    config = load_yaml(ROOT / "configs" / "research" / filename)
    guard_radius = float(
        config["perception"]["scan_guard"]["near_body_stop_radius"]
    )
    audit = audit_reference_path(
        config["scene"],
        config["task"]["points"],
        guard_radius,
        margin=0.03,
        sample_spacing=0.005,
    )

    assert config["plant"]["backend"] == "mujoco_diff_drive"
    assert guard_radius == pytest.approx(0.38)
    assert audit["path_clear"]
    assert audit["minimum_clearance"] >= 0.03
