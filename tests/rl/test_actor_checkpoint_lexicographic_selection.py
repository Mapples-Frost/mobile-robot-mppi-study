import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "experiments" / "rl" / "select_actor_checkpoint_lexicographic.py"
SPEC = importlib.util.spec_from_file_location("actor_selector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _run(tmp_path: Path, seed: int, blocks):
    run_dir = tmp_path / str(seed)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    rows = []
    for step, collision, success, completion in blocks:
        checkpoint_name = "initial.pt" if step == 0 else f"step_{step:09d}.pt"
        (checkpoint_dir / checkpoint_name).write_bytes(f"{seed}:{step}".encode())
        for index in range(2):
            rows.append(
                {
                    "global_step": step,
                    "scene": f"scene_{index}",
                    "seed": seed * 10 + index,
                    "collision": collision,
                    "success": success,
                    "path_completion_ratio": completion,
                    "cross_track_rmse": 0.2,
                    "goal_distance": 1.0,
                    "return": -10.0,
                }
            )
    with (run_dir / "validation_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"git_sha": "abc", "training": {"seed": seed}}), encoding="utf-8"
    )
    (run_dir / "training_summary.json").write_text(
        json.dumps({"global_step": 30_000, "update_records": 29_001}), encoding="utf-8"
    )
    return run_dir


def test_safety_and_success_precede_completion(tmp_path):
    safer = _run(tmp_path, 11, [(0, "False", "False", 0.9)])
    successful = _run(tmp_path, 12, [(0, "False", "True", 0.2)])
    colliding = _run(tmp_path, 13, [(0, "True", "True", 1.0)])

    ranked = MODULE.select([colliding, safer, successful])

    assert ranked[0]["train_seed"] == 12
    assert ranked[1]["train_seed"] == 11
    assert ranked[2]["train_seed"] == 13
    assert ranked[0]["selected"] is True


def test_completion_breaks_equal_safety_success_tie(tmp_path):
    lower = _run(tmp_path, 21, [(5_000, "False", "False", 0.3)])
    higher = _run(tmp_path, 22, [(5_000, "False", "False", 0.5)])

    ranked = MODULE.select([lower, higher])

    assert ranked[0]["train_seed"] == 22
