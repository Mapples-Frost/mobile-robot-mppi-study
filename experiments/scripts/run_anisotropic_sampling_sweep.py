import csv
from itertools import product
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.planners.mppi_receding_horizon_experiment import run_experiment


RESULT_FIELDNAMES = [
    "scene_name",
    "horizon",
    "num_samples",
    "temperature",
    "use_goal_warm_start",
    "use_anisotropic_sampling",
    "sdf_influence_distance",
    "sigma_parallel",
    "sigma_perp",
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
    "use_anisotropic_sampling": False,
    "sdf_influence_distance": 1.2,
    "sigma_parallel": 0.08,
    "sigma_perp": 0.01,
}


SWEEP_CONFIGS = {
    "dense_sampling_ablation_small": {
        "grid_mode": "product",
        "grid": {
            "scene_name": ["dense"],
            "horizon": [15, 35],
            "use_anisotropic_sampling": [False, True],
        },
        "fixed": {
            "num_samples": 250,
            "temperature": 8.0,
            "use_goal_warm_start": False,
            "sdf_influence_distance": 1.2,
            "sigma_parallel": 0.08,
            "sigma_perp": 0.01,
        },
        "table_filename": "dense_sampling_ablation_small_summary.csv",
    },
    "dense_sampling_horizon_compare": {
        "grid_mode": "product",
        "grid": {
            "scene_name": ["dense"],
            "horizon": [10, 15, 20, 25, 30, 35],
            "use_anisotropic_sampling": [False, True],
        },
        "fixed": {
            "num_samples": 250,
            "temperature": 8.0,
            "use_goal_warm_start": False,
            "sdf_influence_distance": 1.2,
            "sigma_parallel": 0.08,
            "sigma_perp": 0.01,
        },
        "table_filename": "dense_sampling_horizon_compare_summary.csv",
    },
    "dense_sampling_sigma_compare_h15": {
        "grid_mode": "zip",
        "grid": {
            "scene_name": ["dense", "dense", "dense", "dense"],
            "use_anisotropic_sampling": [False, True, True, True],
            "sigma_parallel": [0.08, 0.06, 0.08, 0.12],
            "sigma_perp": [0.01, 0.02, 0.01, 0.01],
        },
        "fixed": {
            "horizon": 15,
            "num_samples": 250,
            "temperature": 8.0,
            "use_goal_warm_start": False,
            "sdf_influence_distance": 1.2,
        },
        "table_filename": "dense_sampling_sigma_compare_h15_summary.csv",
    },
    "dense_sampling_sigma_compare_h35": {
        "grid_mode": "zip",
        "grid": {
            "scene_name": ["dense", "dense", "dense", "dense"],
            "use_anisotropic_sampling": [False, True, True, True],
            "sigma_parallel": [0.08, 0.06, 0.08, 0.12],
            "sigma_perp": [0.01, 0.02, 0.01, 0.01],
        },
        "fixed": {
            "horizon": 35,
            "num_samples": 250,
            "temperature": 8.0,
            "use_goal_warm_start": False,
            "sdf_influence_distance": 1.2,
        },
        "table_filename": "dense_sampling_sigma_compare_h35_summary.csv",
    },
}


ACTIVE_SWEEP_NAME = "dense_sampling_horizon_compare"


def build_experiment_cases(base_config, sweep_config):
    """
    根据 sweep 配置，自动展开所有实验组合。

    支持两种模式：
    1. product：笛卡尔积展开
    2. zip：按同位置元素配对展开
    """
    grid = sweep_config["grid"]
    fixed = sweep_config.get("fixed", {})
    grid_mode = sweep_config.get("grid_mode", "product")

    grid_keys = list(grid.keys())
    grid_value_lists = [grid[key] for key in grid_keys]

    experiment_cases = []

    if grid_mode == "product":
        for grid_values in product(*grid_value_lists):
            case = base_config.copy()
            case.update(fixed)

            grid_case_part = dict(zip(grid_keys, grid_values))
            case.update(grid_case_part)

            experiment_cases.append(case)

    elif grid_mode == "zip":
        lengths = [len(v) for v in grid_value_lists]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"zip mode requires all grid lists to have same length, got lengths={lengths}"
            )

        for grid_values in zip(*grid_value_lists):
            case = base_config.copy()
            case.update(fixed)

            grid_case_part = dict(zip(grid_keys, grid_values))
            case.update(grid_case_part)

            experiment_cases.append(case)

    else:
        raise ValueError(
            f"Unknown grid_mode: {grid_mode}. Expected 'product' or 'zip'."
        )

    return experiment_cases


def enrich_result_with_case(result, case):
    """
    把 case 里的关键配置字段补进 result。
    这样即使 run_experiment 暂时没把这些字段写入返回字典，csv 里也能保留下来。
    """
    merged = dict(result)
    for key in RESULT_FIELDNAMES:
        if key in case and key not in merged:
            merged[key] = case[key]
    return merged


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
        result = enrich_result_with_case(result, case)
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
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(all_results)


def print_brief_summary(all_results):
    """
    在终端打印简短摘要，方便快速看对比。
    """
    print()
    print("=" * 108)
    print("Brief Summary")
    print("=" * 108)

    for row in all_results:
        print(
            f'scene={row.get("scene_name")} | '
            f'h={row.get("horizon")} | '
            f'aniso={row.get("use_anisotropic_sampling")} | '
            f'sig_par={row.get("sigma_parallel")} | '
            f'sig_perp={row.get("sigma_perp")} | '
            f'success={row.get("success")} | '
            f'goal_dist={float(row.get("final_goal_distance", 0.0)):.3f} | '
            f'min_clear={float(row.get("min_clearance", 0.0)):.3f} | '
            f'avg_plan_time={float(row.get("average_planning_time", 0.0)):.4f}'
        )


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
    print_brief_summary(all_results)

    table_dir = PROJECT_ROOT / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    csv_path = table_dir / sweep_config["table_filename"]
    save_results_to_csv(all_results, csv_path)

    print()
    print("saved table =", csv_path)


if __name__ == "__main__":
    main()
