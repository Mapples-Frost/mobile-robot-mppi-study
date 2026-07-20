import numpy as np
import pytest

from mobile_robot_mppi.evaluation.tracking import FootprintCorridor


def test_boundary_margin_accounts_for_robot_radius_and_side():
    corridor = FootprintCorridor(half_width=0.85, footprint_radius=0.25)

    left, right = corridor.margins(0.40)

    assert np.isclose(left, 0.20)
    assert np.isclose(right, 1.00)


def test_margin_is_negative_exactly_after_footprint_crosses_boundary():
    corridor = FootprintCorridor(half_width=0.85, footprint_radius=0.25)

    assert min(corridor.margins(0.60)) == pytest.approx(0.0)
    assert min(corridor.margins(0.61)) < 0.0


def test_corridor_rejects_a_footprint_that_cannot_fit():
    with pytest.raises(ValueError, match="footprint must fit"):
        FootprintCorridor(half_width=0.25, footprint_radius=0.25)


def test_variable_width_profile_interpolates_smoothly():
    corridor = FootprintCorridor(
        half_width=0.80,
        footprint_radius=0.20,
        half_width_profile=((0.0, 0.80), (0.4, 0.80), (0.6, 0.725), (1.0, 0.80)),
    )

    assert corridor.half_width_at(0.0) == pytest.approx(0.80)
    assert corridor.half_width_at(0.6) == pytest.approx(0.725)
    assert corridor.half_width_at(0.8) == pytest.approx(0.7625)
