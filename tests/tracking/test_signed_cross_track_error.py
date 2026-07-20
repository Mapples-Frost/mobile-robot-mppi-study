import numpy as np

from mobile_robot_mppi.core.references import PolylineReference
from mobile_robot_mppi.evaluation.tracking import FootprintCorridor, tracking_sample


def test_signed_cross_track_is_positive_on_path_left():
    reference = PolylineReference(((0.0, 0.0), (5.0, 0.0)))
    corridor = FootprintCorridor(half_width=1.0, footprint_radius=0.25)

    left = tracking_sample(reference, (2.0, 0.3, 0.0), corridor)
    right = tracking_sample(reference, (2.0, -0.3, 0.0), corridor)

    assert np.isclose(left.signed_cross_track_error, 0.3)
    assert np.isclose(right.signed_cross_track_error, -0.3)


def test_heading_error_wraps_across_pi_branch_cut():
    reference = PolylineReference(((0.0, 0.0), (-5.0, 0.0)))
    corridor = FootprintCorridor(half_width=1.0, footprint_radius=0.25)

    sample = tracking_sample(reference, (-1.0, 0.0, -np.pi + 0.1), corridor)

    assert np.isclose(sample.tangent_heading_error, 0.1)
