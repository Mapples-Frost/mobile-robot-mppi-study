import csv
from pathlib import Path

from src.planners.mppi_receding_horizon_experiment import run_experiment


def build_experiment_configs():
    """
    返回要做 warm start 对照的实验配置列表。
    每个元素都是一个字典，表示一组固定参数。
    """
    configs = [
        {
            "scene_name": "dense",
            "horizon": 15,
            "num_samples": 250,
            "temperature": 8.0,
            "dt": 0.2,
            "execute_steps": 100,
        },
    ]
    return configs


def run_warm_start_comparison():
    """
    对每一组固定配置，分别跑：
    1. baseline
    2. warm start
    然后把结果收集起来。
    """
    configs = build_experiment_configs()
    rows = []

    for config_idx, config in enumerate(configs, start=1):
        print()
        print("=" * 60)
        print(f"Running config {config_idx}/{len(configs)}")
        print("config =", config)

        for use_goal_warm_start in [False, True]:
            mode_name = "warm_start" if use_goal_warm_start else "baseline"

            print()
            print(f"--- Mode: {mode_name} ---")

            result = run_experiment(
                scene_name=config["scene_name"],
                horizon=config["horizon"],
                num_samples=config["num_samples"],
                temperature=config["temperature"],
                dt=config["dt"],
                execute_steps=config["execute_steps"],
                use_goal_warm_start=use_goal_warm_start,
            )

            row = {
                "comparison_group": (
                    f'{config["scene_name"]}'
                    f'_h{config["horizon"]}'
                    f'_n{config["num_samples"]}'
                    f'_t{config["temperature"]}'
                ),
                "mode_name": mode_name,
                **result,
            }

            rows.append(row)

    return rows


def save_results_to_csv(rows, filename="warm_start_sweep_results.csv"):
    """
    把结果列表写入 csv 文件。
    """
    if not rows:
        print("No rows to save.")
        return

    project_root = Path(__file__).resolve().parents[2]
    table_dir = project_root / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    save_path = table_dir / filename

    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Saved csv =", save_path)


def print_brief_summary(rows):
    """
    在终端里打印一个简短摘要，方便你快速看结果。
    """
    print()
    print("=" * 60)
    print("Brief Summary")
    print("=" * 60)

    for row in rows:
        print(
            f'group={row["comparison_group"]} | '
            f'mode={row["mode_name"]} | '
            f'success={row["success"]} | '
            f'final_goal_distance={row["final_goal_distance"]:.3f} | '
            f'trajectory_length={row["trajectory_length"]:.3f} | '
            f'min_clearance={row["min_clearance"]:.3f} | '
            f'avg_plan_time={row["average_planning_time"]:.4f}'
        )


def main():
    rows = run_warm_start_comparison()
    print_brief_summary(rows)
    save_results_to_csv(rows)


if __name__ == "__main__":
    main()