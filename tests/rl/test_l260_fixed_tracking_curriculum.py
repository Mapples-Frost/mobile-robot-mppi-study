import hashlib
import json
import math
from pathlib import Path

import yaml

from experiments.rl.generate_l260_fixed_tracking_curriculum import (
    BASE_INCLUDE,
    MINIMUM_PATH_CLEARANCE,
    ROBOT_RADIUS,
    TRAIN_SPECS,
    VALIDATION_SPECS,
    generate,
)
from mobile_robot_mppi.core.config import deep_merge, load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import (
    audit_reference_path,
    audit_static_scene,
)


ROOT = Path(__file__).resolve().parents[2]
HELDOUT = (
    "mujoco_tracking_grand_hairpin_l234.yaml",
    "mujoco_tracking_grand_s_chicane_l234.yaml",
    "mujoco_tracking_grand_infinity_l234.yaml",
)


def _points_hash(points):
    payload = json.dumps(points, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_l260_generator_is_deterministic_explicit_and_feasible(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = generate(first)
    generate(second)
    assert manifest["train_count"] == 6
    assert manifest["validation_count"] == 3
    assert len(manifest["maps"]) == 9
    assert [p.relative_to(first) for p in sorted(first.rglob("*.yaml"))] == [
        p.relative_to(second) for p in sorted(second.rglob("*.yaml"))
    ]
    for left in sorted(first.rglob("*.yaml")):
        right = second / left.relative_to(first)
        assert left.read_bytes() == right.read_bytes()
        raw = yaml.safe_load(left.read_text(encoding="utf-8"))
        assert raw["include"] == BASE_INCLUDE
        assert "initial_state" in raw["experiment"]
        assert "obstacles" in raw["scene"]
        assert "field_size" in raw["scene"]
        assert raw["experiment"]["initial_state"][:2] == raw["task"]["points"][0]
        p0, p1 = raw["task"]["points"][:2]
        expected = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        assert raw["experiment"]["initial_state"][2] == expected
        base = load_yaml(ROOT / "configs/research/mujoco_l218_expanded_base.yaml")
        override = dict(raw)
        override.pop("include")
        resolved = deep_merge(base, override)
        assert resolved["experiment"]["initial_state"] == raw["experiment"]["initial_state"]
        assert resolved["scene"]["obstacles"] == raw["scene"]["obstacles"]
        radius = resolved["plant"]["robot"]["collision_radius"]
        assert radius == ROBOT_RADIUS
        path = audit_reference_path(
            resolved["scene"], resolved["task"]["points"], radius,
            margin=MINIMUM_PATH_CLEARANCE, sample_spacing=0.01,
        )
        connectivity = audit_static_scene(
            resolved["scene"], resolved["experiment"]["initial_state"][:2],
            resolved["task"]["points"][-1], radius,
            margin=MINIMUM_PATH_CLEARANCE, resolution=0.05,
        )
        assert path["path_clear"], (left.name, path)
        assert connectivity["start_free"], left.name
        assert connectivity["goal_free"], left.name
        assert connectivity["path_exists"], left.name


def test_l260_source_geometry_is_unique_and_not_exact_heldout():
    specs = tuple(TRAIN_SPECS) + tuple(VALIDATION_SPECS)
    hashes = [_points_hash(spec["points"]) for spec in specs]
    assert len(hashes) == len(set(hashes))
    heldout_hashes = {
        _points_hash(load_yaml(ROOT / "configs/research" / name)["task"]["points"])
        for name in HELDOUT
    }
    assert heldout_hashes.isdisjoint(hashes)
    text = json.dumps(specs, sort_keys=True).lower()
    for forbidden in ("hairpin_mountain_pass", "sinusoidal_s_chicane", "gerono_infinity_with_exit", "sealed"):
        assert forbidden not in text


def test_l260_manifest_paths_and_roles_are_isolated(tmp_path):
    manifest = generate(tmp_path)
    roles = [record["role"] for record in manifest["maps"]]
    assert roles.count("train") == 6
    assert roles.count("validation") == 3
    assert len({record["geometry_sha256"] for record in manifest["maps"]}) == 9
    for record in manifest["maps"]:
        path = Path(record["path"])
        assert path.exists()
        assert record["minimum_footprint_clearance"] >= MINIMUM_PATH_CLEARANCE
        assert ("/train/" in path.as_posix()) == (record["role"] == "train")


def test_checked_in_l260_package_matches_generator_and_resolves_explicitly(tmp_path):
    generated = tmp_path / "generated"
    expected_manifest = generate(generated)
    checked_in = ROOT / "configs/research/l260_curriculum"
    actual_manifest = json.loads(
        (checked_in / "manifest.json").read_text(encoding="utf-8")
    )
    def without_locations(manifest):
        result = json.loads(json.dumps(manifest))
        for record in result["maps"]:
            record.pop("path")
        return result

    assert without_locations(actual_manifest) == without_locations(expected_manifest)
    for record in actual_manifest["maps"]:
        checked_path = ROOT / record["path"]
        generated_path = generated / Path(record["path"]).relative_to(
            "configs/research/l260_curriculum"
        )
        assert checked_path.read_bytes() == generated_path.read_bytes()
        raw = yaml.safe_load(checked_path.read_text(encoding="utf-8"))
        resolved = load_yaml(checked_path)
        assert resolved["experiment"]["initial_state"] == raw["experiment"]["initial_state"]
        assert resolved["task"]["points"] == raw["task"]["points"]
        assert resolved["scene"]["field_size"] == raw["scene"]["field_size"]
        assert resolved["scene"]["obstacles"] == raw["scene"]["obstacles"]
