import copy
import csv
import json
from pathlib import Path

from experiments.rl.summarize_high_dynamic_data_gate import summarize
from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_high_dynamic_data_gate_checks_artifacts_and_excitation(tmp_path):
    config = copy.deepcopy(load_yaml(
        ROOT / "configs/rl/icode_high_dynamic_data_l56.yaml"
    ))
    design = config["rl"]["cross_layer_factorial"]
    design["development_episode_seeds"] = [101]
    gate = config["residual_dataset"]["data_gate"]
    gate["expected_episodes"] = 4
    block = tmp_path / "block_0"
    condition = "traditional_nominal"
    domain = "fixed_hidden_mismatch_l56"
    episode_rows, step_rows = [], []
    scene_names = []
    state_values = ((0.0, -0.5), (0.2, 0.5), (0.4, -0.5), (0.6, 0.5))
    for index, item in enumerate(design["scenes"]):
        scene_config = load_yaml(ROOT / item["path"])
        scene = scene_config["scene"]["name"]
        scene_names.append(scene)
        episode_rows.append({
            "scene": scene,
            "physics_domain": domain,
            "episode_seed": 101,
            "condition": condition,
            "success": True,
            "collision": False,
            "steps": 1,
        })
        state_v, state_omega = state_values[index]
        step_rows.append({
            "scene": scene,
            "physics_domain": domain,
            "episode_seed": 101,
            "condition": condition,
            "step": 0,
            "executed_v": 0.5,
            "executed_omega": state_omega,
            "applied_v": 0.5,
            "applied_omega": state_omega,
            "v": state_v,
            "omega": state_omega,
            "safety_override": 0.0,
        })
        run = block / "runs" / scene / domain / condition / "seed_101"
        run.mkdir(parents=True, exist_ok=True)
        (run / "config_resolved.yaml").write_text(json.dumps({
            name: scene_config[name] for name in (
                "experiment", "task", "state_space", "action_space",
                "plant", "planner", "sensors", "rl",
            )
        }), encoding="utf-8")
    _write_csv(block / "episodes.csv", episode_rows)
    _write_csv(block / "factorial_steps.csv", step_rows)
    (block / "metadata.json").write_text(json.dumps({
        "model_block": 0,
        "episode_seeds": [101],
        "conditions": [condition],
        "sealed_confirmation_seeds_used": [],
        "previous_protected_seeds_used": [],
        "episodes": 4,
    }), encoding="utf-8")

    result = summarize(config, tmp_path)

    assert result["gate_passed"], result
    assert result["integrity"]["plant_contract_count"] == 1
    assert result["integrity"]["isolated_sensor_contract"]
    assert {row["scene"] for row in result["per_scene"]} == set(scene_names)
