import pytest

from experiments.icode.collect_value_alignment_data import (
    _environment_signature,
    _selected_environments,
)


def _environment(scene, domain, role):
    return {
        "scene": {"name": scene},
        "experiment": {
            "physics_domain": domain,
            "physics_domain_role": role,
        },
    }


def test_path_reliability_split_filters_scene_prefix_and_domain_role():
    snapshot = {
        "training_environments": [
            _environment("straight__nominal", "nominal", "seen"),
            _environment("straight__combined", "combined", "unseen"),
            _environment("sweep__nominal", "nominal", "seen"),
        ],
        "validation_environments": [],
    }

    selected = _selected_environments(
        snapshot,
        "training",
        ("seen",),
        ("straight",),
    )

    assert [_environment_signature(item) for item in selected] == [
        ("straight__nominal", "nominal")
    ]


def test_path_reliability_split_fails_closed_when_filter_is_empty():
    snapshot = {
        "training_environments": [
            _environment("straight__nominal", "nominal", "seen"),
        ],
        "validation_environments": [],
    }

    with pytest.raises(ValueError, match="scene prefixes"):
        _selected_environments(
            snapshot,
            "training",
            ("seen",),
            ("hairpin",),
        )
