from types import SimpleNamespace

import numpy as np

from mobile_robot_mppi.real_robot.human_point_cloud import (
    HumanPointCloudEvidenceExtractor,
)


def _observation(points):
    return SimpleNamespace(
        pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        auxiliary={"human_point_cloud_base": np.asarray(points, dtype=float)},
    )


def _track(x=1.5, y=0.0):
    return {
        "track_index": 0,
        "associated": True,
        "measurement_x": x,
        "measurement_y": y,
    }


def _body_points(center_x=1.5, center_y=0.0):
    points = []
    for height in (0.20, 0.35, 0.75, 0.95, 1.35, 1.55):
        for dx in (-0.12, 0.0, 0.12):
            for dy in (-0.16, 0.0, 0.16):
                points.append((center_x + dx, center_y + dy, height))
    return points


def test_full_height_body_evidence_annotates_existing_track():
    extractor = HumanPointCloudEvidenceExtractor({"enabled": True})
    tracks, diagnostics = extractor.annotate(
        (_track(),), _observation(_body_points())
    )
    assert diagnostics["candidate_track_indices"] == (0,)
    assert tracks[0]["point_cloud_human_candidate"] is True
    assert tracks[0]["point_cloud_human_shape"] == "full_body"
    assert tracks[0]["point_cloud_human_occupied_height_bands"] == 3
    assert tracks[0]["point_cloud_human_vertical_span_m"] > 1.0


def test_wide_vertical_wall_patch_is_not_human_shape():
    extractor = HumanPointCloudEvidenceExtractor({"enabled": True})
    points = [
        (1.5, y, height)
        for y in np.linspace(-0.62, 0.62, 21)
        for height in (0.20, 0.80, 1.40)
    ]
    tracks, diagnostics = extractor.annotate(
        (_track(),), _observation(points)
    )
    assert diagnostics["candidate_track_indices"] == ()
    assert tracks[0]["point_cloud_human_candidate"] is False
    assert tracks[0]["point_cloud_human_horizontal_extent_m"] > 0.95


def test_point_cloud_evidence_never_creates_a_track_slot():
    extractor = HumanPointCloudEvidenceExtractor({"enabled": True})
    tracks, diagnostics = extractor.annotate((), _observation(_body_points()))
    assert tracks == []
    assert diagnostics["evaluated_track_count"] == 0
    assert diagnostics["candidate_track_indices"] == ()
