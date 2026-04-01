import csv
from itertools import product
from pathlib import Path

from src.planners.mppi_receding_horizon_experiment import run_experiment


RESULT_FIELDNAMES = [
    "scene_name",
    "horizon",
    "num_samples",
    "temperature",
    "dt",
    "execute_steps",
    "executed_trajectory_points",
    "trajectory_length",
    "success",
    "final_goal_distance",
    "min_clearance",
    "total_planning_time",
    "average_planning_time",
    "final_executed_cost",
    "final_collision",
    "final_state",
]


BASE_EXPERIMENT_CONFIG = {
    "num_samples": 250,
    "temperature": 8.0,
    "dt": 0.2,
    "execute_steps": 100,
    "v_std": 0.25,
    "omega_std": 0.35,
    "v_min": 0.0,
    "v_max": 1.4,
    "omega_max": 1.2,
    "goal_tolerance": 0.35,
    "robot_radius": 0.25,
    "use_goal_warm_start": False,
}


SWEEP_CONFIGS = {
    "cross_scene_horizon": {
        "grid": {
            "scene_name": ["sparse", "dense", "narrow"],
            "horizon": [15, 25, 35],
        },
        "fixed": {},
        "table_filename": "cross_scene_horizon_summary.csv",
    },
    "dense_num_samples": {
        "grid": {
            "scene_name": ["dense"],
            "horizon": [35],
            "num_samples": [250, 400, 600],
        },
        "fixed": {},
        "table_filename": "dense_h35_num_samples_summary.csv",
    },
    "dense_temperature": {
        "grid": {
            "scene_name": ["dense"],
            "horizon": [35],
            "temperature": [4.0, 8.0, 12.0],
        },
        "fixed": {},
        "table_filename": "dense_h35_temperature_summary.csv",
    },
}


ACTIVE_SWEEP_NAME = "dense_temperature"


def build_experiment_cases(base_config, sweep_config):
    """
    根据 sweep 配置，自动展开所有实验组合。
    """
    grid = sweep_config["grid"]
    fixed = sweep_config.get("fixed", {})

    grid_keys = list(grid.keys())
    grid_value_lists = [grid[key] for key in grid_keys]

    experiment_cases = []

    for grid_values in product(*grid_value_lists):
        case = base_config.copy()
        case.update(fixed)

        grid_case_part = dict(zip(grid_keys, grid_values))
        case.update(grid_case_part)

        experiment_cases.append(case)

    return experiment_cases


def run_cases(experiment_cases):
    """
    逐个运行实验 case，收集结果。
    """
    all_results = []

    for case_idx, case in enumerate(experiment_cases, start=1):
        print("\n" + "=" * 72)
        print(f"Running case {case_idx}/{len(experiment_cases)}")
        for key, value in case.items():
            print(f"{key} = {value}")
        print("=" * 72)

        result = run_experiment(**case)
        all_results.append(result)

    return all_results


def save_results_to_csv(all_results, csv_path):
    """
    把实验结果写入 csv。
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=RESULT_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(all_results)


def main():
    if ACTIVE_SWEEP_NAME not in SWEEP_CONFIGS:
        raise ValueError(
            f"Unknown ACTIVE_SWEEP_NAME: {ACTIVE_SWEEP_NAME}. "
            f"Available options: {list(SWEEP_CONFIGS.keys())}"
        )

    sweep_config = SWEEP_CONFIGS[ACTIVE_SWEEP_NAME]
    experiment_cases = build_experiment_cases(
        base_config=BASE_EXPERIMENT_CONFIG,
        sweep_config=sweep_config,
    )

    all_results = run_cases(experiment_cases)

    project_root = Path(__file__).resolve().parents[2]
    table_dir = project_root / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    csv_path = table_dir / sweep_config["table_filename"]
    save_results_to_csv(all_results, csv_path)

    print()
    print("saved table =", csv_path)


if __name__ == "__main__":
    main()