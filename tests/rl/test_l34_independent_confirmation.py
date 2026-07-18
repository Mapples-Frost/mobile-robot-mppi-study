from pathlib import Path

import pytest

from experiments.rl.run_l34_independent_confirmation import (
    _confirmation_design,
    _schedule,
)
from experiments.rl.summarize_l34_independent_confirmation import (
    _exact_two_sided_sign_p,
    _paired_rows,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "rl" / "l34_independent_confirmation_l35.yaml"


def test_l35_design_is_frozen_disjoint_and_balanced():
    _, design = _confirmation_design(CONFIG)
    assert design["fixed_alpha"] == 0.25
    assert design["prediction_mode"] == "nominal"
    assert set(design["episode_seeds"]).isdisjoint(
        set(design["forbidden_episode_seeds"])
    )
    assert len(design["training_runs"]) == 3
    assert len(design["scenes"]) == 2
    schedule = _schedule(design)
    assert len(schedule) == 12
    assert len(schedule) * len(design["episode_seeds"]) == 60


def test_l35_pairing_rejects_missing_checkpoint_role():
    rows = [{
        "training_seed": "1",
        "scene": "s",
        "seed": "9",
        "checkpoint_role": "initial",
        "success": "False",
        "collision": "False",
        "final_goal_distance": "1.0",
    }]
    with pytest.raises(ValueError, match="missing initial/best"):
        _paired_rows(rows)


def test_exact_sign_test_is_two_sided_and_handles_no_discordance():
    assert _exact_two_sided_sign_p(6, 0) == pytest.approx(0.03125)
    assert _exact_two_sided_sign_p(3, 3) == pytest.approx(1.0)
    assert _exact_two_sided_sign_p(0, 0) is None

