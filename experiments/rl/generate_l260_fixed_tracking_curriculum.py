"""Generate the fixed, geometry-audited L260 Tracking curriculum.

Unlike L258, every generated scene explicitly owns its initial state, path,
field, and obstacles.  The generator refuses to write a scene unless the
robot footprint can traverse the entire reference with a frozen clearance
margin.  No held-out Tracking geometry is read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from mobile_robot_mppi.evaluation.scene_feasibility import (
    audit_reference_path,
    audit_static_scene,
)


PROTOCOL = "L260"
FIELD_SIZE = (8.0, 8.0)
ROBOT_RADIUS = 0.25
MINIMUM_PATH_CLEARANCE = 0.08
BOUNDARY_RESERVE = ROBOT_RADIUS + MINIMUM_PATH_CLEARANCE
BASE_INCLUDE = "../../mujoco_l218_expanded_base.yaml"


def _cylinder(x, y, radius=0.28):
    return {
        "type": "cylinder",
        "position": [float(x), float(y)],
        "radius": float(radius),
        "height": 0.32,
    }


def _box(x, y, sx, sy, yaw=0.0):
    result = {
        "type": "box",
        "position": [float(x), float(y)],
        "size": [float(sx), float(sy)],
        "height": 0.32,
    }
    if yaw:
        result["yaw"] = float(yaw)
    return result


# These are fixed development-training templates, not samples from the final
# Hairpin, S-Chicane, or Infinity geometries.  Keeping them fixed makes the
# 10k-versus-20k comparison interpretable.
TRAIN_SPECS = (
    {
        "name": "l260_train_s_bend",
        "family": "s_bend",
        "points": [[0.9, 1.2], [1.8, 1.35], [2.8, 2.1], [3.8, 3.25],
                   [4.8, 3.45], [5.8, 2.55], [7.0, 2.25]],
        "obstacles": [_cylinder(2.0, 3.2), _cylinder(4.6, 1.7),
                      _cylinder(6.4, 3.55)],
    },
    {
        "name": "l260_train_offset_hairpin",
        "family": "offset_hairpin",
        "points": [[0.9, 0.9], [6.8, 0.9], [6.8, 2.1], [1.5, 2.5],
                   [1.2, 3.5], [6.6, 4.0], [6.7, 5.2], [1.0, 6.3]],
        "obstacles": [_box(3.9, 1.65, 2.25, 0.10),
                      _box(4.5, 3.00, 2.05, 0.10),
                      _box(3.9, 4.75, 2.10, 0.10)],
    },
    {
        "name": "l260_train_near_double_loop",
        "family": "near_double_loop",
        "points": [[0.9, 4.0], [1.7, 2.4], [3.2, 1.3], [5.2, 1.4],
                   [6.7, 2.7], [7.0, 4.4], [5.9, 6.2], [4.1, 6.7],
                   [2.2, 6.0], [1.1, 4.5], [2.4, 3.3], [4.1, 3.0],
                   [5.8, 3.5], [6.6, 4.8], [5.2, 5.4], [3.8, 4.7],
                   [2.5, 5.1]],
        "obstacles": [_cylinder(0.75, 1.1, 0.24),
                      _cylinder(7.15, 6.9, 0.24)],
    },
    {
        "name": "l260_train_curvature_ramp",
        "family": "curvature_ramp",
        "points": [[0.9, 0.9], [1.9, 0.95], [2.9, 1.15], [3.9, 1.6],
                   [4.8, 2.35], [5.55, 3.35], [6.0, 4.55],
                   [6.05, 5.8], [5.55, 6.9]],
        "obstacles": [_cylinder(3.0, 2.25), _cylinder(5.0, 4.25),
                      _cylinder(7.0, 5.4)],
    },
    {
        "name": "l260_train_corridor_switchback",
        "family": "corridor_switchback",
        "points": [[0.9, 1.0], [6.8, 1.0], [6.8, 2.35], [1.3, 2.8],
                   [1.2, 4.15], [6.7, 4.6], [6.8, 6.5]],
        # Keep a deliberate buffer above the frozen 0.08 m footprint
        # clearance threshold.  A previous 0.10 m half-width technically
        # passed by only 0.012 m, which is too close for a training fixture.
        "obstacles": [_box(3.8, 1.72, 2.25, 0.04),
                      _box(3.8, 3.48, 2.05, 0.04),
                      _box(4.0, 5.35, 2.15, 0.04)],
    },
    {
        "name": "l260_train_compound_turns",
        "family": "compound_turns",
        "points": [[0.9, 6.8], [1.4, 5.7], [2.4, 5.0], [3.5, 5.2],
                   [4.2, 4.3], [3.8, 3.2], [4.6, 2.25], [5.8, 2.5],
                   [6.9, 1.3]],
        "obstacles": [_cylinder(1.5, 4.35), _cylinder(3.0, 3.8),
                      _cylinder(5.4, 3.8), _cylinder(6.6, 2.7)],
    },
)


VALIDATION_SPECS = (
    {
        "name": "l260_validation_wave",
        "family": "validation_wave",
        "points": [[0.8, 2.0], [1.8, 2.6], [2.9, 3.8], [4.0, 4.15],
                   [5.1, 3.45], [6.1, 2.25], [7.1, 1.8]],
        "obstacles": [_cylinder(2.4, 1.7), _cylinder(4.4, 5.15),
                      _cylinder(6.4, 3.25)],
    },
    {
        "name": "l260_validation_switchback",
        "family": "validation_switchback",
        "points": [[1.0, 0.8], [1.1, 6.8], [2.55, 6.8], [3.0, 1.35],
                   [4.4, 1.1], [4.9, 6.65], [6.4, 6.8], [7.0, 1.0]],
        "obstacles": [_box(1.82, 4.0, 0.10, 2.1),
                      _box(3.7, 3.9, 0.10, 2.0),
                      _box(5.7, 4.0, 0.10, 2.0)],
    },
    {
        "name": "l260_validation_loop_exit",
        "family": "validation_loop_exit",
        "points": [[0.9, 3.8], [1.6, 2.2], [3.1, 1.2], [5.0, 1.4],
                   [6.4, 2.7], [6.6, 4.5], [5.4, 5.9], [3.7, 6.4],
                   [2.1, 5.7], [1.25, 4.4], [2.7, 3.55], [4.5, 3.75],
                   [6.0, 4.9], [7.1, 6.4]],
        "obstacles": [_cylinder(0.75, 6.8, 0.24),
                      _cylinder(7.15, 1.0, 0.24)],
    },
)


def _initial_state(points):
    (x0, y0), (x1, y1) = points[:2]
    return [float(x0), float(y0), float(math.atan2(y1 - y0, x1 - x0)), 0.0, 0.0]


def _scene_document(spec, role):
    points = [[float(x), float(y)] for x, y in spec["points"]]
    return {
        "include": BASE_INCLUDE,
        "experiment": {
            "name": spec["name"],
            "initial_state": _initial_state(points),
            "max_steps": 1400,
            "physics_domain_role": "seen" if role == "train" else "unseen",
        },
        "task": {
            "type": "polyline",
            "points": points,
        },
        "scene": {
            "name": spec["name"],
            "geometry_id": spec["name"],
            "field_size": list(FIELD_SIZE),
            "obstacles": list(spec["obstacles"]),
        },
    }


def _canonical_geometry(document):
    geometry = {
        "initial_state": document["experiment"]["initial_state"],
        "points": document["task"]["points"],
        "field_size": document["scene"]["field_size"],
        "obstacles": document["scene"]["obstacles"],
    }
    return json.dumps(geometry, sort_keys=True, separators=(",", ":"))


def _audit_document(document):
    points = document["task"]["points"]
    initial = document["experiment"]["initial_state"]
    if [float(v) for v in initial[:2]] != [float(v) for v in points[0]]:
        raise ValueError("initial XY must exactly equal the first reference point")
    expected_yaw = math.atan2(
        points[1][1] - points[0][1], points[1][0] - points[0][0]
    )
    if abs(math.atan2(math.sin(initial[2] - expected_yaw), math.cos(initial[2] - expected_yaw))) > 1e-10:
        raise ValueError("initial yaw must align with the first reference segment")
    width, height = document["scene"]["field_size"]
    for x, y in points:
        if not (
            BOUNDARY_RESERVE <= x <= width - BOUNDARY_RESERVE
            and BOUNDARY_RESERVE <= y <= height - BOUNDARY_RESERVE
        ):
            raise ValueError("reference violates the footprint boundary reserve")
    reference = audit_reference_path(
        document["scene"], points, ROBOT_RADIUS,
        margin=MINIMUM_PATH_CLEARANCE, sample_spacing=0.01,
    )
    connectivity = audit_static_scene(
        document["scene"], initial[:2], points[-1], ROBOT_RADIUS,
        margin=MINIMUM_PATH_CLEARANCE, resolution=0.05,
    )
    if not reference["path_clear"]:
        raise ValueError(
            "reference intersects inflated obstacle geometry: %s"
            % reference["minimum_clearance"]
        )
    if not (
        connectivity["start_free"]
        and connectivity["goal_free"]
        and connectivity["path_exists"]
    ):
        raise ValueError("scene start/goal connectivity audit failed")
    return {"reference": reference, "connectivity": connectivity}


def generate(output_root: Path):
    output_root = Path(output_root)
    records = []
    fingerprints = set()
    for role, specs in (("train", TRAIN_SPECS), ("validation", VALIDATION_SPECS)):
        directory = output_root / role
        directory.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            document = _scene_document(spec, role)
            audit = _audit_document(document)
            fingerprint = hashlib.sha256(
                _canonical_geometry(document).encode("utf-8")
            ).hexdigest()
            if fingerprint in fingerprints:
                raise ValueError("duplicate L260 geometry fingerprint")
            fingerprints.add(fingerprint)
            path = directory / f"{spec['name']}.yaml"
            path.write_text(
                "# l260_fixed_geometry: explicit start/path/field/obstacles\n"
                + yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            records.append({
                "role": role,
                "name": spec["name"],
                "family": spec["family"],
                "path": path.as_posix(),
                "geometry_sha256": fingerprint,
                "reference_length": audit["reference"]["reference_length"],
                "minimum_footprint_clearance": audit["reference"]["minimum_clearance"],
                "initial_state": document["experiment"]["initial_state"],
                "obstacle_count": len(document["scene"]["obstacles"]),
            })
    manifest = {
        "protocol": PROTOCOL,
        "generator": Path(__file__).name,
        "design": "fixed_maps_with_blocked_seed_and_physics_randomization",
        "field_size": list(FIELD_SIZE),
        "robot_radius": ROBOT_RADIUS,
        "minimum_path_clearance": MINIMUM_PATH_CLEARANCE,
        "train_count": len(TRAIN_SPECS),
        "validation_count": len(VALIDATION_SPECS),
        "maps": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
