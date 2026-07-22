import numpy as np

from experiments.rl.generate_l268_heldout_recovery_diagnostic import (
    STUDENT_FIELDS,
    _action_rows,
)


def test_l268_heldout_actions_are_fixed_and_distinct_semantics():
    rows = _action_rows(
        np.asarray((0.25, -0.50), dtype=np.float32),
        np.asarray((-0.10, 0.75), dtype=np.float32),
    )
    assert tuple(row[0] for row in rows) == (
        "recorded_recovery", "source_actor", "fast_forward"
    )
    assert np.array_equal(rows[-1][1], np.asarray((1.0, 0.0), dtype=np.float32))


def test_l268_student_fields_exclude_privileged_truth():
    assert STUDENT_FIELDS == {
        "observations", "actions", "rewards", "constraint_costs",
        "next_observations", "dones", "groups", "chain_ids", "steps",
    }
    assert not ({"truth_pose", "cte_m", "path_progress_m"} & STUDENT_FIELDS)
