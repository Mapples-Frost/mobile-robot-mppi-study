"""Regression tests for ``known_static_map_segment_half_thickness``.

Two requirements:

1. **Default is unchanged.** ``known_static_map_segment_half_thickness = False``
   must reproduce the historical full-thickness clearance exactly, so every
   frozen protocol and sealed result stays bit-reproducible.

2. **Corrected mode matches the plant.** With the flag enabled, the planner's
   segment clearance must equal the geometry MuJoCo actually builds
   (``model_factory.py`` half-extent ``0.5 * thickness``) and the convention
   used by ``mujoco_plant._minimum_clearance``, ``scene_feasibility`` and
   ``static_astar``.

The second is the point of the change: the planner was over-conservative by
``0.5 * thickness`` on every segment obstacle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mobile_robot_mppi.core.spaces import ActionSpec, StateSpec
from mobile_robot_mppi.planning.dynamics import LegacyUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController

ROBOT_RADIUS = 0.25

# Thickness values taken from the chapter-1 map (36 segments).
SEGMENTS = [
    {"type": "segment", "start": [0.0, 0.0], "end": [2.0, 0.0], "thickness": 0.20},
    {"type": "segment", "start": [0.0, 1.5], "end": [2.0, 1.5], "thickness": 0.28},
    {"type": "segment", "start": [-1.0, -1.0], "end": [-1.0, 1.0], "thickness": 0.22},
]


def _controller(half_thickness: bool) -> MppiController:
    return MppiController(
        dynamics=LegacyUnicyclePrediction(),
        config=MppiConfig(
            horizon=4,
            num_samples=8,
            dt=0.1,
            robot_radius=ROBOT_RADIUS,
            known_static_map_segment_half_thickness=half_thickness,
        ),
        state_spec=StateSpec(names=("x", "y", "theta"), periodic_indices=(2,)),
        action_spec=ActionSpec(
            names=("v_cmd", "omega_cmd"),
            lower=np.array([-0.35, -0.9]),
            upper=np.array([0.7, 0.95]),
        ),
    )


def _plant_convention(point, obstacles, robot_radius=ROBOT_RADIUS):
    """Reimplements mujoco_plant._minimum_clearance for segment geometry.

    MuJoCo builds a segment as a box with half-extent 0.5 * thickness
    (model_factory.py:47), so this is ground truth.
    """
    best = math.inf
    for o in obstacles:
        a = np.asarray(o["start"], dtype=np.float64)
        b = np.asarray(o["end"], dtype=np.float64)
        ab = b - a
        den = float(ab @ ab)
        t = 0.0 if den <= 1e-12 else float(np.clip((point - a) @ ab / den, 0.0, 1.0))
        d = float(np.linalg.norm(point - (a + t * ab)))
        best = min(best, d - 0.5 * float(o["thickness"]))
    return best - robot_radius


def _legacy_convention(point, obstacles, robot_radius=ROBOT_RADIUS):
    """Reimplements the pre-change planner expression (full thickness)."""
    best = math.inf
    for o in obstacles:
        a = np.asarray(o["start"], dtype=np.float64)
        b = np.asarray(o["end"], dtype=np.float64)
        ab = b - a
        den = float(ab @ ab)
        t = 0.0 if den <= 1e-12 else float(np.clip((point - a) @ ab / den, 0.0, 1.0))
        d = float(np.linalg.norm(point - (a + t * ab)))
        best = min(best, d - float(o["thickness"]))
    return best - robot_radius


PROBE_POINTS = [
    (0.5, 0.6), (1.0, 0.75), (1.5, 0.9), (-0.4, 0.0),
    (0.2, 1.1), (1.8, 0.35), (-0.8, 0.5), (0.9, 0.45),
]


def _trajectory(points):
    return np.array([[[x, y, 0.0] for x, y in points]], dtype=np.float64)


class TestDefaultUnchanged:
    def test_config_default_is_legacy(self):
        assert MppiConfig().known_static_map_segment_half_thickness is False

    def test_from_mapping_default_is_legacy(self):
        assert (
            MppiConfig.from_mapping({}, 2).known_static_map_segment_half_thickness
            is False
        )

    def test_from_mapping_accepts_override(self):
        cfg = MppiConfig.from_mapping(
            {"known_static_map_segment_half_thickness": True}, 2
        )
        assert cfg.known_static_map_segment_half_thickness is True

    def test_default_reproduces_full_thickness_exactly(self):
        controller = _controller(half_thickness=False)
        got = controller._known_static_map_clearance(
            _trajectory(PROBE_POINTS), SEGMENTS
        )[0]
        want = np.array([_legacy_convention(np.array(p), SEGMENTS) for p in PROBE_POINTS])
        assert np.allclose(got, want, rtol=0.0, atol=1e-12), (
            "default mode diverged from the historical full-thickness expression"
        )


class TestCorrectedMatchesPlant:
    def test_corrected_equals_plant_convention_exactly(self):
        controller = _controller(half_thickness=True)
        got = controller._known_static_map_clearance(
            _trajectory(PROBE_POINTS), SEGMENTS
        )[0]
        want = np.array([_plant_convention(np.array(p), SEGMENTS) for p in PROBE_POINTS])
        assert np.allclose(got, want, rtol=0.0, atol=1e-12), (
            "corrected mode does not match the geometry MuJoCo builds"
        )

    def test_correction_recovers_exactly_half_thickness(self):
        legacy = _controller(False)._known_static_map_clearance(
            _trajectory(PROBE_POINTS), SEGMENTS
        )[0]
        fixed = _controller(True)._known_static_map_clearance(
            _trajectory(PROBE_POINTS), SEGMENTS
        )[0]
        gain = fixed - legacy
        # Each probe point's nearest segment may differ, so the gain is
        # 0.5 * thickness of whichever segment binds.
        allowed = {0.5 * o["thickness"] for o in SEGMENTS}
        for value in gain:
            assert any(abs(value - a) < 1e-9 for a in allowed), (
                f"unexpected clearance gain {value}"
            )
        assert np.all(gain > 0.0), "correction must never reduce clearance"

    def test_corrected_is_never_more_permissive_than_truth(self):
        """The fix must not make the planner optimistic relative to the plant."""
        controller = _controller(half_thickness=True)
        got = controller._known_static_map_clearance(
            _trajectory(PROBE_POINTS), SEGMENTS
        )[0]
        truth = np.array([_plant_convention(np.array(p), SEGMENTS) for p in PROBE_POINTS])
        assert np.all(got <= truth + 1e-12)


class TestOtherGeometryUntouched:
    @pytest.mark.parametrize(
        "obstacle",
        [
            {"type": "cylinder", "position": [1.0, 1.0], "radius": 0.3},
            {"type": "box", "position": [1.0, 1.0], "size": [0.3, 0.2], "yaw": 0.4},
        ],
    )
    def test_flag_does_not_affect_box_or_cylinder(self, obstacle):
        legacy = _controller(False)._known_static_map_clearance(
            _trajectory(PROBE_POINTS), [obstacle]
        )
        fixed = _controller(True)._known_static_map_clearance(
            _trajectory(PROBE_POINTS), [obstacle]
        )
        assert np.array_equal(legacy, fixed), (
            "the segment flag must not change box or cylinder geometry"
        )
