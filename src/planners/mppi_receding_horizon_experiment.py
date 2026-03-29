"""
最小 2D MPPI receding horizon demo（重构版）

这版重构主要做了 4 件事：
1. 从 experiments/configs/baseline_scenes.py 读取场景。
2. 给每个场景增加 bounds（工作空间边界）。
3. 在 trajectory_cost(...) 里增加边界惩罚，避免机器人钻地图外圈漏洞。
4. 每个场景单独保存一张图到 results/figures/。
"""

import math
import random
from pathlib import Path
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import experiments.configs.baseline_scenes as baseline_scenes


def wrap_angle(theta):
    """把角度规范到 [-pi, pi]。"""
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def step(state, control, dt):
    """
    单步状态更新。

    state = (x, y, theta)
    control = (v, omega)
    dt = 时间步长
    """
    x, y, theta = state
    v, omega = control

    next_x = x + v * math.cos(theta) * dt
    next_y = y + v * math.sin(theta) * dt
    next_theta = wrap_angle(theta + omega * dt)

    return next_x, next_y, next_theta


def rollout_control_sequence(start_state, control_sequence, dt):
    """
    从起点状态出发，把一整条控制序列 rollout 成一条轨迹。
    """
    trajectory = [start_state]
    current_state = start_state

    for control in control_sequence:
        current_state = step(current_state, control, dt)
        trajectory.append(current_state)

    return trajectory


def initialize_control_sequence(nominal_control, horizon):
    """生成初始参考控制序列。"""
    return [nominal_control for _ in range(horizon)]

def build_goal_warm_start_sequence(
    current_state,
    goal,
    horizon,
    dt,
    v_max,
    omega_max,
):
    warm_start_sequence = []
    rollout_state = current_state

    for _ in range(horizon):
        x, y, theta = rollout_state
        goal_x, goal_y = goal

        desired_heading = math.atan2(goal_y - y, goal_x - x)
        heading_error = wrap_angle(desired_heading - theta)

        omega = max(-omega_max, min(omega_max, heading_error / max(dt, 1e-8)))

        if abs(heading_error) > 0.6:
            v = 0.4 * v_max
        elif abs(heading_error) > 0.25:
            v = 0.7 * v_max
        else:
            v = 1.0 * v_max

        control = (v, omega)
        warm_start_sequence.append(control)
        rollout_state = step(rollout_state, control, dt)

    return warm_start_sequence


def sample_control_sequences(
    nominal_sequence,
    num_samples,
    v_std,
    omega_std,
    v_min,
    v_max,
    omega_max,
):
    """
    围绕 nominal_sequence 采样出很多条候选控制序列。
    """
    sampled_sequences = []

    for _ in range(num_samples):
        sampled_sequence = []

        for nominal_v, nominal_omega in nominal_sequence:
            sampled_v = random.gauss(nominal_v, v_std)
            sampled_omega = random.gauss(nominal_omega, omega_std)

            sampled_v = max(v_min, min(v_max, sampled_v))
            sampled_omega = max(-omega_max, min(omega_max, sampled_omega))

            sampled_sequence.append((sampled_v, sampled_omega))

        sampled_sequences.append(sampled_sequence)

    if sampled_sequences:
        sampled_sequences[0] = nominal_sequence.copy()

    return sampled_sequences


def obstacle_clearance(x, y, obstacle, robot_radius):
    """
    计算机器人当前位置到某个圆形障碍物边界的净空距离。
    obstacle = (obs_x, obs_y, obs_radius)
    """
    obs_x, obs_y, obs_radius = obstacle
    center_distance = math.hypot(x - obs_x, y - obs_y)
    clearance = center_distance - (robot_radius + obs_radius)
    return clearance


def boundary_penalty(x, y, bounds):
    """
    边界惩罚：
    - 出界：重罚
    - 虽未出界但贴边：轻罚
    """
    x_min = bounds["x_min"]
    x_max = bounds["x_max"]
    y_min = bounds["y_min"]
    y_max = bounds["y_max"]

    if x < x_min or x > x_max or y < y_min or y > y_max:
        out_dx = max(x_min - x, 0.0) + max(x - x_max, 0.0)
        out_dy = max(y_min - y, 0.0) + max(y - y_max, 0.0)
        return 5000.0 + 2000.0 * (out_dx ** 2 + out_dy ** 2)

    margin = min(x - x_min, x_max - x, y - y_min, y_max - y)
    if margin < 0.35:
        return 40.0 * (0.35 - margin) ** 2

    return 0.0


def trajectory_cost(trajectory, control_sequence, goal, obstacles, robot_radius, bounds):
    """
    计算一条轨迹的总代价。

    当前代价包括：
    1. 每一步离目标太远
    2. 每一步朝向误差
    3. 每一步角速度过大
    4. 靠障碍太近 / 碰撞
    5. 靠地图边界太近 / 出界
    6. 最终终点距离和终点朝向误差
    7. 控制变化不平滑
    """
    goal_x, goal_y = goal
    total_cost = 0.0
    collided = False

    for i, state in enumerate(trajectory[1:], start=1):
        x, y, theta = state

        # 1. 每一步离目标太远
        goal_distance = math.hypot(goal_x - x, goal_y - y)
        total_cost += 0.15 * goal_distance

        # 2. 每一步朝向误差
        desired_heading = math.atan2(goal_y - y, goal_x - x)
        heading_error = abs(wrap_angle(desired_heading - theta))
        total_cost += 0.05 * heading_error

        # 3. 每一步角速度过大
        _, omega = control_sequence[i - 1]
        total_cost += 0.01 * (omega ** 2)

        # 4. 靠障碍太近 / 碰撞
        for obstacle in obstacles:
            clearance = obstacle_clearance(x, y, obstacle, robot_radius)

            if clearance <= 0.0:
                collided = True
                total_cost += 10000.0
            elif clearance < 0.35:
                total_cost += 20.0 * (0.8 - clearance) ** 2

        # 5. 边界惩罚
        total_cost += boundary_penalty(x, y, bounds)

    # 6. 最终终点距离和终点朝向误差
    final_x, final_y, final_theta = trajectory[-1]
    final_goal_distance = math.hypot(goal_x - final_x, goal_y - final_y)
    total_cost += 100.0 * final_goal_distance

    final_desired_heading = math.atan2(goal_y - final_y, goal_x - final_x)
    final_heading_error = abs(wrap_angle(final_desired_heading - final_theta))
    total_cost += 2.0 * final_heading_error

    # 7. 控制变化不平滑
    for i in range(1, len(control_sequence)):
        prev_v, prev_omega = control_sequence[i - 1]
        curr_v, curr_omega = control_sequence[i]
        total_cost += 0.08 * ((curr_v - prev_v) ** 2 + 0.2 * (curr_omega - prev_omega) ** 2)

    return total_cost, collided


def compute_weights(costs, temperature):
    """根据每条候选轨迹的代价计算权重。"""
    min_cost = min(costs)
    shifted_costs = [cost - min_cost for cost in costs]

    unnormalized_weights = [
        math.exp(-shifted_cost / max(temperature, 1e-8))
        for shifted_cost in shifted_costs
    ]

    weight_sum = sum(unnormalized_weights)
    if weight_sum < 1e-12:
        return [1.0 / len(costs)] * len(costs)

    return [w / weight_sum for w in unnormalized_weights]


def weighted_update_sequence(sampled_sequences, weights):
    """按权重把很多条候选控制序列加权平均，得到新的参考控制序列。"""
    horizon = len(sampled_sequences[0])
    updated_sequence = []

    for t in range(horizon):
        weighted_v = 0.0
        weighted_omega = 0.0

        for sampled_sequence, weight in zip(sampled_sequences, weights):
            v, omega = sampled_sequence[t]
            weighted_v += weight * v
            weighted_omega += weight * omega

        updated_sequence.append((weighted_v, weighted_omega))

    return updated_sequence


def shift_sequence(sequence, tail_control=(0.0, 0.0)):
    """把控制序列整体向前滚一格。"""
    if not sequence:
        return []

    shifted = sequence[1:].copy()
    shifted.append(tail_control)
    return shifted


def executed_trajectory_cost(executed_trajectory, executed_controls, goal, obstacles, robot_radius, bounds):
    """单独给真实执行出来的轨迹算总代价。"""
    return trajectory_cost(
        executed_trajectory,
        executed_controls,
        goal,
        obstacles,
        robot_radius,
        bounds,
    )

def compute_trajectory_length(trajectory):
    """计算真实执行轨迹的总路径长度。"""
    total_length = 0.0

    for i in range(1, len(trajectory)):
        prev_x, prev_y, _ = trajectory[i - 1]
        curr_x, curr_y, _ = trajectory[i]

        segment_length = math.hypot(curr_x - prev_x, curr_y - prev_y)
        total_length += segment_length

    return total_length


def compute_min_clearance(trajectory, obstacles, robot_radius):
    """计算真实执行轨迹全过程中的最小障碍净空。"""
    min_clearance = float("inf")

    for state in trajectory:
        x, y, _ = state

        for obstacle in obstacles:
            clearance = obstacle_clearance(x, y, obstacle, robot_radius)
            if clearance < min_clearance:
                min_clearance = clearance

    return min_clearance


def plot_result(
    executed_trajectory,
    sampled_best_trajectory,
    start_state,
    goal,
    obstacles,
    robot_radius,
    scene_name,
    bounds,
    horizon=None,
    num_samples=None,
    temperature=None,
):
    """画图并保存结果。"""
    fig, ax = plt.subplots(figsize=(8, 6))

    start_x, start_y, _ = start_state
    goal_x, goal_y = goal

    ax.scatter(start_x, start_y, s=80, label="start")
    ax.scatter(goal_x, goal_y, s=80, label="goal")

    if sampled_best_trajectory is not None:
        best_xs = [state[0] for state in sampled_best_trajectory]
        best_ys = [state[1] for state in sampled_best_trajectory]
        ax.plot(best_xs, best_ys, "--", label="last best predicted trajectory")

    executed_xs = [state[0] for state in executed_trajectory]
    executed_ys = [state[1] for state in executed_trajectory]
    ax.plot(executed_xs, executed_ys, linewidth=2.5, label="executed trajectory")

    for obs_x, obs_y, obs_radius in obstacles:
        obstacle_circle = Circle((obs_x, obs_y), obs_radius, fill=False, linewidth=2)
        ax.add_patch(obstacle_circle)

    final_x, final_y, _ = executed_trajectory[-1]
    robot_circle = Circle((final_x, final_y), robot_radius, fill=False, linestyle="--")
    ax.add_patch(robot_circle)

    ax.set_title(f"MPPI Receding Horizon Demo - {scene_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(bounds["x_min"], bounds["x_max"])
    ax.set_ylim(bounds["y_min"], bounds["y_max"])
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    project_root = Path(__file__).resolve().parents[2]
    figure_dir = project_root / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if horizon is None:
        save_path = figure_dir / f"mppi_{scene_name}.png"
    elif num_samples is None:
        save_path = figure_dir / f"mppi_{scene_name}_h{horizon}.png"
    elif temperature is None:
        save_path = figure_dir / f"mppi_{scene_name}_h{horizon}_n{num_samples}.png"
    else:
        save_path = figure_dir / f"mppi_{scene_name}_h{horizon}_n{num_samples}_t{temperature}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print("saved figure =", save_path)

def run_experiment(
    scene_name,
    horizon,
    num_samples=250,
    temperature=8.0,
    dt=0.2,
    execute_steps=100,
    v_std=0.25,
    omega_std=0.35,
    v_min=0.0,
    v_max=1.4,
    omega_max=1.2,
    goal_tolerance=0.35,
    robot_radius=0.25,
    use_goal_warm_start=False,
):
    random.seed(11)

    print("loaded baseline_scenes file =", baseline_scenes.__file__)
    print("available scene keys =", baseline_scenes.SCENES.keys())

    scene = baseline_scenes.SCENES[scene_name]
    start_state = scene["start_state"]
    goal = scene["goal"]
    obstacles = scene["obstacles"]
    bounds = scene["bounds"]

    nominal_control = (1.0, 0.0)
    if use_goal_warm_start:
        nominal_sequence = build_goal_warm_start_sequence(
            current_state=start_state,
            goal=goal,
            horizon=horizon,
            dt=dt,
            v_max=v_max,
            omega_max=omega_max,
        )
    else:
        nominal_sequence = initialize_control_sequence(nominal_control, horizon)

    current_state = start_state
    executed_trajectory = [current_state]
    executed_controls = []
    sampled_best_trajectory = None
    total_planning_time = 0.0
    planning_steps = 0

    print("=== MPPI Receding Horizon Demo ===")
    print("start_state =", start_state)
    print("goal =", goal)
    print("scene_name =", scene_name)
    print("bounds =", bounds)
    print("horizon =", horizon)
    print("num_samples =", num_samples)
    print("temperature =", temperature)
    print()

    for step_idx in range(execute_steps):
        step_plan_start = time.perf_counter()
        sampled_sequences = sample_control_sequences(
            nominal_sequence=nominal_sequence,
            num_samples=num_samples,
            v_std=v_std,
            omega_std=omega_std,
            v_min=v_min,
            v_max=v_max,
            omega_max=omega_max,
        )

        sampled_trajectories = []
        costs = []
        collisions = []

        for control_sequence in sampled_sequences:
            trajectory = rollout_control_sequence(current_state, control_sequence, dt)
            total_cost, collided = trajectory_cost(
                trajectory=trajectory,
                control_sequence=control_sequence,
                goal=goal,
                obstacles=obstacles,
                robot_radius=robot_radius,
                bounds=bounds,
            )

            sampled_trajectories.append(trajectory)
            costs.append(total_cost)
            collisions.append(collided)

        weights = compute_weights(costs, temperature)
        updated_sequence = weighted_update_sequence(sampled_sequences, weights)

        step_plan_end = time.perf_counter()
        step_planning_time = step_plan_end - step_plan_start
        total_planning_time += step_planning_time
        planning_steps += 1

        best_idx = min(range(len(costs)), key=lambda i: costs[i])
        sampled_best_trajectory = sampled_trajectories[best_idx]
        best_cost = costs[best_idx]
        best_collision = collisions[best_idx]

        executed_control = updated_sequence[0]
        current_state = step(current_state, executed_control, dt)

        executed_controls.append(executed_control)
        executed_trajectory.append(current_state)

        nominal_sequence = shift_sequence(updated_sequence, tail_control=(0.0, 0.0))

        goal_distance = math.hypot(goal[0] - current_state[0], goal[1] - current_state[1])

        print(
            f"Step {step_idx:02d} | "
            f"best_cost = {best_cost:.3f} | "
            f"best_collision = {best_collision} | "
            f"executed_control = ({executed_control[0]:.3f}, {executed_control[1]:.3f}) | "
            f"current_state = ({current_state[0]:.3f}, {current_state[1]:.3f}, {current_state[2]:.3f}) | "
            f"goal_distance = {goal_distance:.3f}"
        )

        if goal_distance < goal_tolerance:
            print()
            print(f"Reached goal-distance tolerance at step {step_idx:02d}.")
            print("Final success will still be determined by collision check and final metrics.")
            break

    final_cost, final_collision = executed_trajectory_cost(
        executed_trajectory=executed_trajectory,
        executed_controls=executed_controls,
        goal=goal,
        obstacles=obstacles,
        robot_radius=robot_radius,
        bounds=bounds,
    )

    final_state = executed_trajectory[-1]
    final_goal_distance = math.hypot(goal[0] - final_state[0], goal[1] - final_state[1])
    success = (final_goal_distance < goal_tolerance) and (not final_collision)

    trajectory_length = compute_trajectory_length(executed_trajectory)
    min_clearance = compute_min_clearance(executed_trajectory, obstacles, robot_radius)

    average_planning_time = total_planning_time / max(planning_steps, 1)

    print()
    print("=== Final Result ===")
    print("Success =", success)
    print(f"Final goal distance = {final_goal_distance:.3f}")
    print(f"Trajectory length = {trajectory_length:.3f}")
    print(f"Min clearance = {min_clearance:.3f}")
    print(f"Total planning time = {total_planning_time:.4f} s")
    print(f"Average planning time = {average_planning_time:.4f} s")
    print("Executed trajectory points =", len(executed_trajectory))
    print(f"Final executed cost = {final_cost:.3f}")
    print("Final collision =", final_collision)
    print("Final state =", final_state)

    plot_result(
        executed_trajectory=executed_trajectory,
        sampled_best_trajectory=sampled_best_trajectory,
        start_state=start_state,
        goal=goal,
        obstacles=obstacles,
        robot_radius=robot_radius,
        scene_name=scene_name,
        bounds=bounds,
        horizon=horizon,
        num_samples=num_samples,
        temperature=temperature,
    )

    result = {
        "scene_name": scene_name,
        "horizon": horizon,
        "num_samples": num_samples,
        "temperature": temperature,
        "dt": dt,
        "execute_steps": execute_steps,
        "executed_trajectory_points": len(executed_trajectory),
        "trajectory_length": trajectory_length,
        "success": success,
        "final_goal_distance": final_goal_distance,
        "min_clearance": min_clearance,
        "total_planning_time": total_planning_time,
        "average_planning_time": average_planning_time,
        "final_executed_cost": final_cost,
        "final_collision": final_collision,
        "final_state": final_state,
    }

    return result

def main():
    result = run_experiment(
        scene_name="dense",
        horizon=35,
        num_samples=250,
        temperature=8.0,
        dt=0.2,
        execute_steps=100,
        use_goal_warm_start=False,
    )

    print()
    print("=== Main Wrapper Result ===")
    print(result)


if __name__ == "__main__":
    main()
