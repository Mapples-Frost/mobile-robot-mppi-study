import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import math
import random
from pathlib import Path
import numpy as np
import time
from src.envs.mujoco_point_env import MujocoPointEnv

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
    warm_start_prefix_steps=8,
    nominal_tail_control=(1.0, 0.0),
):
    warm_start_sequence = []
    rollout_state = current_state

    effective_prefix_steps = min(horizon, warm_start_prefix_steps)# 热启动前缀步数

    for _ in range(effective_prefix_steps):
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

    remaining_steps = horizon - effective_prefix_steps
    for _ in range(remaining_steps):
        warm_start_sequence.append(nominal_tail_control)

    return warm_start_sequence


def sample_control_sequences(
    nominal_sequence,
    current_state,
    dt,
    obstacles,
    robot_radius,
    num_samples,
    v_std,
    omega_std,
    v_min,
    v_max,
    omega_max,
    use_anisotropic_sampling=False,#anisotropic 各向异性
    sdf_influence_distance=1.2,#几何采样引导距离
    sigma_parallel=0.08,#切向扩散强度
    sigma_perp=0.01,#法向扩散强度
    goal=None,#goal 坐标，后面用来判断“哪边切向更朝目标”
    bounds=None,#边界，后面用来算 inward bias
    isotropic_anchor_ratio=0.25,#保留 25% 的各向同性样本，防止全体 sample 一起塌缩
    tangent_push_gain=0.12,#给 goal-aligned 切向一个轻微前推
    boundary_bias_distance=0.45#距离边界 0.45m 内，就开始给 inward bias
    ,boundary_push_gain=0.90#界内推强度
):
    """
    围绕 nominal_sequence 采样很多条候选控制序列。
    当 use_anisotropic_sampling=True 时：
    - 先根据 nominal rollout 得到每个 t 的 nominal state
    - 再用最近障碍的 SDF 梯度构造位置相关协方差
    """
    sampled_sequences = []

    nominal_trajectory = rollout_control_sequence(
        start_state=current_state,
        control_sequence=nominal_sequence,
        dt=dt,
    )

    anisotropic_meta = None
    zero_mean_xy = np.zeros(2, dtype=float)
    if use_anisotropic_sampling:
        anisotropic_meta = []

        # Cache nominal-rollout geometry once; it does not depend on sample index.
        for t, (nominal_v, nominal_omega) in enumerate(nominal_sequence):
            x_t, y_t, theta_t = nominal_trajectory[t]
            clearance_t, grad_t = nearest_obstacle_sdf_gradient(
                x=x_t,
                y=y_t,
                obstacles=obstacles,
                robot_radius=robot_radius,
            )

            step_meta = {
                "clearance_t": clearance_t,
                "theta_t": theta_t,
                "nominal_v": nominal_v,
                "nominal_omega": nominal_omega,
            }

            if clearance_t <= sdf_influence_distance:
                n = grad_t / max(np.linalg.norm(grad_t), 1e-8)
                t_dir = np.array([-n[1], n[0]], dtype=float)

                u_nom_xy = np.array(
                    [
                        nominal_v * math.cos(theta_t),
                        nominal_v * math.sin(theta_t),
                    ],
                    dtype=float,
                )

                if goal is not None:
                    goal_dir = np.array([goal[0] - x_t, goal[1] - y_t], dtype=float)
                    goal_dir_norm = np.linalg.norm(goal_dir)

                    if goal_dir_norm > 1e-8:
                        goal_dir = goal_dir / goal_dir_norm

                        if np.dot(t_dir, goal_dir) < 0.0:
                            t_dir = -t_dir

                tangent_bias_xy = tangent_push_gain * t_dir

                boundary_bias_xy = np.zeros(2, dtype=float)
                if bounds is not None:
                    boundary_margin, inward_normal = nearest_boundary_inward_normal(
                        x=x_t,
                        y=y_t,
                        bounds=bounds,
                        robot_radius=robot_radius,
                    )

                    if boundary_margin < boundary_bias_distance:
                        boundary_strength = boundary_push_gain * (
                            boundary_bias_distance - boundary_margin
                        )
                        boundary_bias_xy = boundary_strength * inward_normal

                step_meta["Sigma"] = (
                    sigma_parallel * np.outer(t_dir, t_dir)
                    + sigma_perp * np.outer(n, n)
                )
                step_meta["u_nom_xy"] = u_nom_xy
                step_meta["tangent_bias_xy"] = tangent_bias_xy
                step_meta["boundary_bias_xy"] = boundary_bias_xy

            anisotropic_meta.append(step_meta)

    for _ in range(num_samples):
        sampled_sequence = []

        for t, (nominal_v, nominal_omega) in enumerate(nominal_sequence):

            if not use_anisotropic_sampling:
                sampled_v = random.gauss(nominal_v, v_std)
                sampled_omega = random.gauss(nominal_omega, omega_std)

            else:
                step_meta = anisotropic_meta[t]
                clearance_t = step_meta["clearance_t"]
                theta_t = step_meta["theta_t"]
                nominal_v = step_meta["nominal_v"]
                nominal_omega = step_meta["nominal_omega"]

                # 先留一部分各向同性 anchor，防止 sample cloud 一起塌向同一边
                if (
                        clearance_t > sdf_influence_distance
                        or random.random() < isotropic_anchor_ratio
                ):
                    sampled_v = random.gauss(nominal_v, v_std)
                    sampled_omega = random.gauss(nominal_omega, omega_std)
                else:
                    eps_xy = np.random.multivariate_normal(
                        mean=zero_mean_xy,
                        cov=step_meta["Sigma"],
                    )

                    # 最终真正参与采样的二维速度向量
                    u_sample_xy = (
                            step_meta["u_nom_xy"]
                            + step_meta["tangent_bias_xy"]
                            + step_meta["boundary_bias_xy"]
                            + eps_xy
                    )





                    sampled_v = float(np.linalg.norm(u_sample_xy))
                    sampled_v = max(v_min, min(v_max, sampled_v))

                    if sampled_v < 1e-8:
                        desired_heading = theta_t
                    else:
                        desired_heading = math.atan2(
                            u_sample_xy[1],
                            u_sample_xy[0],
                        )

                        """
                          新版逻辑：
                          nominal_sequence⟶nominal_trajectory⟶(xt​,yt​,θt​)⟶(clearancet​,gradt​)⟶(n,t)⟶Σ⟶ϵxy​∼N(0,Σ)⟶usample,xy​⟶(sampledv​,sampledω​)
                          即，先根据nominal sequnce算出一个轨迹，然后从第一步开始，计算每一步的clearance and grand，
                          然后根据这两个数据构造协方差矩阵，再在xy方向上加一个
                          加权后的噪声，由于此时的噪声是一个二维空间里的速度向量扰动
                          因此需要先把这一步的线速度v转换成一个速度向量，再对速度向量加噪声
                           最后反解出加噪声后的v和Omega构造sample sequence
                         """

                    sampled_omega = wrap_angle(desired_heading - theta_t) / max(dt, 1e-8)

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


def nearest_obstacle_sdf_gradient(x, y, obstacles, robot_radius):
    """
    sdf:signed distance field 带符号距离场
    gradientL梯度
    返回：
    1. 到最近障碍边界的 signed distance / clearance
    2. 该位置的近似 SDF 梯度方向（单位向量，指向远离障碍的法向）
    """
    min_clearance = float("inf")
    best_grad = np.array([1.0, 0.0], dtype=float)

    for obs_x, obs_y, obs_radius in obstacles:
        dx = x - obs_x
        dy = y - obs_y
        center_distance = math.hypot(dx, dy)
        clearance = center_distance - (robot_radius + obs_radius)

        if clearance < min_clearance:
            min_clearance = clearance

            if center_distance > 1e-8:
                best_grad = np.array(
                    [dx / center_distance, dy / center_distance],
                    dtype=float,
                )
            else:
                best_grad = np.array([1.0, 0.0], dtype=float)

    return min_clearance, best_grad


def nearest_geometry_sdf_gradient(x, y, obstacles, bounds, robot_radius):
    """
    返回当前位置相对于“最近几何体”的 clearance 和法向梯度。
    这里的几何体包括：
    1. 圆形障碍物
    2. 四条有效边界（考虑 robot_radius 之后）

    返回：
    - min_clearance: 到最近几何约束的净空
    - best_grad: 对应的外法向方向（单位向量）
    """
    # 先用“最近障碍”初始化
    min_clearance, best_grad = nearest_obstacle_sdf_gradient(
        x=x,
        y=y,
        obstacles=obstacles,
        robot_radius=robot_radius,
    )

    effective_x_min = bounds["x_min"] + robot_radius
    effective_x_max = bounds["x_max"] - robot_radius
    effective_y_min = bounds["y_min"] + robot_radius
    effective_y_max = bounds["y_max"] - robot_radius

    boundary_candidates = [
        (x - effective_x_min, np.array([1.0, 0.0], dtype=float)),   # 左边界
        (effective_x_max - x, np.array([-1.0, 0.0], dtype=float)),  # 右边界
        (y - effective_y_min, np.array([0.0, 1.0], dtype=float)),   # 下边界
        (effective_y_max - y, np.array([0.0, -1.0], dtype=float)),  # 上边界
    ]

    for clearance, grad in boundary_candidates:
        if clearance < min_clearance:
            min_clearance = clearance
            best_grad = grad

    return min_clearance, best_grad

def boundary_penalty(x, y, bounds, robot_radius):
    """
    边界惩罚：
    - 把边界当成硬约束，机器人本体不能出界
    - 未出界但贴边时，提前给更强惩罚
    """
    x_min = bounds["x_min"]
    x_max = bounds["x_max"]
    y_min = bounds["y_min"]
    y_max = bounds["y_max"]

    safe_buffer = 0.10

    effective_x_min = x_min + robot_radius
    effective_x_max = x_max - robot_radius
    effective_y_min = y_min + robot_radius
    effective_y_max = y_max - robot_radius

    if (
        x < effective_x_min
        or x > effective_x_max
        or y < effective_y_min
        or y > effective_y_max
    ):
        out_dx = max(effective_x_min - x, 0.0) + max(x - effective_x_max, 0.0)
        out_dy = max(effective_y_min - y, 0.0) + max(y - effective_y_max, 0.0)
        return 20000.0 + 8000.0 * (out_dx ** 2 + out_dy ** 2)

    margin = min(
        x - effective_x_min,
        effective_x_max - x,
        y - effective_y_min,
        effective_y_max - y,
    )

    warning_margin = robot_radius + safe_buffer
    if margin < warning_margin:
        return 300.0 * (warning_margin - margin) ** 2

    return 0.0


def nearest_boundary_inward_normal(x, y, bounds, robot_radius):
    effective_x_min = bounds["x_min"] + robot_radius
    effective_x_max = bounds["x_max"] - robot_radius
    effective_y_min = bounds["y_min"] + robot_radius
    effective_y_max = bounds["y_max"] - robot_radius

    candidates = [
        (x - effective_x_min, np.array([1.0, 0.0], dtype=float)),   # 左边界，向右推
        (effective_x_max - x, np.array([-1.0, 0.0], dtype=float)),  # 右边界，向左推
        (y - effective_y_min, np.array([0.0, 1.0], dtype=float)),   # 下边界，向上推
        (effective_y_max - y, np.array([0.0, -1.0], dtype=float)),  # 上边界，向下推
    ]

    min_margin, inward_normal = min(candidates, key=lambda item: item[0])
    return min_margin, inward_normal


def is_inside_effective_bounds(state, bounds, robot_radius):
    """
    检查机器人圆盘本体是否仍在有效边界内。
    这里检查的是机器人中心点 + robot_radius 之后的可行区域。
    """
    x, y, _ = state

    effective_x_min = bounds["x_min"] + robot_radius
    effective_x_max = bounds["x_max"] - robot_radius
    effective_y_min = bounds["y_min"] + robot_radius
    effective_y_max = bounds["y_max"] - robot_radius

    return (
        effective_x_min <= x <= effective_x_max
        and effective_y_min <= y <= effective_y_max
    )


def would_step_out_of_bounds(state, control, dt, bounds, robot_radius):
    """
    先做一步前向预测，判断执行这个 control 后会不会出有效边界。
    """
    next_state = step(state, control, dt)
    return not is_inside_effective_bounds(next_state, bounds, robot_radius)


def make_boundary_safe_control(
    current_state,
    proposed_control,
    dt,
    bounds,
    robot_radius,
):
    """
    对 planner 给出的 executed_control 做最后一道边界安全过滤。

    逻辑：
    1. 先尝试原始控制；
    2. 如果单步会出界，则逐步缩小线速度 v；
    3. 如果还不安全，则返回 (0.0, 0.0) 停住。

    返回：
    - safe_control: 最终执行控制
    - intervention: 这一步是否发生了安全干预
    """
    proposed_v, proposed_omega = proposed_control

    candidate_controls = [
        (proposed_v, proposed_omega),
        (0.75 * proposed_v, proposed_omega),
        (0.50 * proposed_v, proposed_omega),
        (0.25 * proposed_v, proposed_omega),
        (0.10 * proposed_v, proposed_omega),
        (0.00, proposed_omega),
        (0.00, 0.00),
    ]

    for candidate_control in candidate_controls:
        if not would_step_out_of_bounds(
            state=current_state,
            control=candidate_control,
            dt=dt,
            bounds=bounds,
            robot_radius=robot_radius,
        ):
            intervention = (candidate_control != proposed_control)
            return candidate_control, intervention

    return (0.0, 0.0), True



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

    progress_cost = 0.0 #推进代价，鼓励朝着终点走
    safety_cost = 0.0 #安全代价，罚过于贴近边界和碰撞
    smoothness_cost = 0.0 #平滑代价，罚一次变化过大
    terminal_cost = 0.0 #终点代价，罚离中带你太远

    collided = False
    horizon_steps = max(len(control_sequence), 1)#horizon总步数

    for i, state in enumerate(trajectory[1:], start=1):
        x, y, theta = state

        # 1. 每一步离目标太远
        goal_distance = math.hypot(goal_x - x, goal_y - y)
        progress_cost += 0.15 * goal_distance

        # 2. 每一步朝向误差
        desired_heading = math.atan2(goal_y - y, goal_x - x)
        heading_error = abs(wrap_angle(desired_heading - theta))
        progress_cost += 0.05 * heading_error

        # 3. 每一步角速度过大
        _, omega = control_sequence[i - 1]
        progress_cost += 0.01 * (omega ** 2)

        # 4. 靠障碍太近 / 碰撞
        safe_clearance = 0.70
        danger_clearance = 0.20

        for obstacle in obstacles:
            clearance = obstacle_clearance(x, y, obstacle, robot_radius)

            if clearance <= 0.0:
                collided = True
                safety_cost += 20000.0
                break

            if clearance < safe_clearance:
                safety_cost += 80.0 * (safe_clearance - clearance) ** 2

            if clearance < danger_clearance:
                safety_cost += 400.0 * (danger_clearance - clearance) ** 2

        # 5. 边界惩罚
        boundary_cost = boundary_penalty(x, y, bounds, robot_radius)
        safety_cost += boundary_cost
        if boundary_cost >= 20000.0:
            collided = True
            break

    # 6. 最终终点距离和终点朝向误差
    final_x, final_y, final_theta = trajectory[-1]
    final_goal_distance = math.hypot(goal_x - final_x, goal_y - final_y)
    terminal_cost += 100.0 * final_goal_distance

    final_desired_heading = math.atan2(goal_y - final_y, goal_x - final_x)
    final_heading_error = abs(wrap_angle(final_desired_heading - final_theta))
    terminal_cost += 2.0 * final_heading_error

    final_min_clearance = float("inf")
    for obstacle in obstacles:
        clearance = obstacle_clearance(final_x, final_y, obstacle, robot_radius)
        if clearance < final_min_clearance:
            final_min_clearance = clearance

    # 原有的终点净空惩罚：保留
    if final_min_clearance < 0.70:
        terminal_cost += 120.0 * (0.70 - final_min_clearance) ** 2

    # 新增：goal 附近更严格的终点安全排序
    if final_goal_distance < 1.0:
        desired_terminal_clearance = 0.95
        if final_min_clearance < desired_terminal_clearance:
            terminal_cost += 280.0 * (desired_terminal_clearance - final_min_clearance) ** 2

    # 7. 控制变化不平滑
    for i in range(1, len(control_sequence)):
        prev_v, prev_omega = control_sequence[i - 1]
        curr_v, curr_omega = control_sequence[i]
        smoothness_cost += 0.08 * (
            (curr_v - prev_v) ** 2 + 0.2 * (curr_omega - prev_omega) ** 2
        )

    total_cost = (
        (progress_cost + smoothness_cost) / horizon_steps
        + safety_cost
        + terminal_cost
    )

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

def select_top_k_trajectories(sampled_trajectories, costs, k=5):
    """
    从当前这一步的所有候选轨迹里，选出 cost 最小的前 k 条轨迹。
    返回值是一个 trajectory 列表。
    """
    if not sampled_trajectories or not costs:
        return []

    sorted_indices = sorted(range(len(costs)), key=lambda i: costs[i])
    top_k_indices = sorted_indices[:k]

    top_k_trajectories = [sampled_trajectories[i] for i in top_k_indices]
    return top_k_trajectories



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
    use_goal_warm_start=False,
    use_anisotropic_sampling=False,   # 新增
    top_k_first_step_trajectories=None,
    top_k_last_step_trajectories=None,
    sdf_influence_distance=None,
    sigma_parallel=None,
    sigma_perp=None,
):
    """画图并保存结果。"""
    fig, ax = plt.subplots(figsize=(8, 6))

    start_x, start_y, _ = start_state
    goal_x, goal_y = goal

    ax.scatter(start_x, start_y, s=80, label="start")
    ax.scatter(goal_x, goal_y, s=80, label="goal")

    if top_k_first_step_trajectories is not None:
        for traj_idx, trajectory in enumerate(top_k_first_step_trajectories):
            xs = [state[0] for state in trajectory]
            ys = [state[1] for state in trajectory]

            if traj_idx == 0:
                ax.plot(xs, ys, color="gray", alpha=0.25, linewidth=1.2, label="top-k at first step")
            else:
                ax.plot(xs, ys, color="gray", alpha=0.25, linewidth=1.2)

    if top_k_last_step_trajectories is not None:
        for traj_idx, trajectory in enumerate(top_k_last_step_trajectories):
            xs = [state[0] for state in trajectory]
            ys = [state[1] for state in trajectory]

            if traj_idx == 0:
                ax.plot(xs, ys, color="green", alpha=0.25, linewidth=1.2, label="top-k at last step")
            else:
                ax.plot(xs, ys, color="green", alpha=0.25, linewidth=1.2)

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

    ax.set_title(
        f"MPPI - {scene_name} | "
        f"h={horizon} | "
        f"t={temperature} | "
        f"sdf={sdf_influence_distance} | "
        f"aniso={use_anisotropic_sampling}"
    )
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

    warm_tag = "warm" if use_goal_warm_start else "baseline"
    sampling_tag = "anisotropic" if use_anisotropic_sampling else "isotropic"
    mode_tag = f"{warm_tag}_{sampling_tag}"

    if horizon is None:
        save_path = figure_dir / f"mppi_{scene_name}_{mode_tag}.png"
    elif num_samples is None:
        save_path = figure_dir / f"mppi_{scene_name}_h{horizon}_{mode_tag}.png"
    elif temperature is None:
        save_path = figure_dir / f"mppi_{scene_name}_h{horizon}_n{num_samples}_{mode_tag}.png"
    else:
        save_path = figure_dir / (
            f"mppi_{scene_name}"
            f"_h{horizon}"
            f"_n{num_samples}"
            f"_t{temperature}"
            f"_sdf{sdf_influence_distance}"
            f"_{mode_tag}.png"
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print("saved figure =", save_path)

def run_experiment_mujoco(
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
    use_anisotropic_sampling=False,
    sdf_influence_distance=1.2,
    sigma_parallel=0.08,
    sigma_perp=0.01,
):
    random.seed(11)
    np.random.seed(11)

    print("loaded baseline_scenes file =", baseline_scenes.__file__)
    print("available scene keys =", baseline_scenes.SCENES.keys())

    scene = baseline_scenes.SCENES[scene_name]
    start_state = scene["start_state"]
    goal = scene["goal"]
    obstacles = scene["obstacles"]
    bounds = scene["bounds"]

    project_root = Path(__file__).resolve().parents[2]
    xml_path = project_root / "src" / "models" / "mujoco" / "scene_minimal_robot.xml"

    env = MujocoPointEnv(xml_path=xml_path)
    env.launch_viewer()
    env.reset(start_state)


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
    top_k_first_step_trajectories = None
    top_k_last_step_trajectories = None
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
            current_state=current_state,
            dt=dt,
            obstacles=obstacles,
            robot_radius=robot_radius,
            num_samples=num_samples,
            v_std=v_std,
            omega_std=omega_std,
            v_min=v_min,
            v_max=v_max,
            omega_max=omega_max,
            use_anisotropic_sampling=use_anisotropic_sampling,
            sdf_influence_distance=sdf_influence_distance,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            goal=goal,
            bounds=bounds,
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


        current_top_k_trajectories = select_top_k_trajectories(
            sampled_trajectories=sampled_trajectories,
            costs=costs,
            k=5,
        )

        aux_predicted_trajectories = current_top_k_trajectories[1:3]

        env.update_best_predicted_trajectory(sampled_best_trajectory)
        env.update_aux_predicted_trajectories(aux_predicted_trajectories)




        if step_idx == 0:
            top_k_first_step_trajectories = current_top_k_trajectories

        top_k_last_step_trajectories = current_top_k_trajectories

        proposed_control = updated_sequence[0]
        executed_control, safety_intervened = make_boundary_safe_control(
            current_state=current_state,
            proposed_control=proposed_control,
            dt=dt,
            bounds=bounds,
            robot_radius=robot_radius,
        )

        current_state = env.step(executed_control, dt)
        env.render()

        executed_controls.append(executed_control)
        executed_trajectory.append(current_state)


        tail_control = updated_sequence[-1]
        nominal_sequence = shift_sequence(updated_sequence, tail_control=tail_control)

        goal_distance = math.hypot(goal[0] - current_state[0], goal[1] - current_state[1])

        print(
            f"Step {step_idx:02d} | "
            f"best_cost = {best_cost:.3f} | "
            f"best_collision = {best_collision} | "
            f"safety_intervened = {safety_intervened} | "
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
        use_goal_warm_start=use_goal_warm_start,
        use_anisotropic_sampling=use_anisotropic_sampling,  # 新增
        top_k_first_step_trajectories=top_k_first_step_trajectories,
        top_k_last_step_trajectories=top_k_last_step_trajectories,
        sdf_influence_distance=sdf_influence_distance,
        sigma_parallel=sigma_parallel,
        sigma_perp=sigma_perp,
    )

    result = {
        "scene_name": scene_name,
        "horizon": horizon,
        "num_samples": num_samples,
        "temperature": temperature,
        "use_goal_warm_start": use_goal_warm_start,
        "use_anisotropic_sampling": use_anisotropic_sampling,  # 新增
        "sdf_influence_distance": sdf_influence_distance,  # 新增
        "sigma_parallel": sigma_parallel,
        "sigma_perp": sigma_perp,
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
    env.close()

    return result



def main():
    result = run_experiment_mujoco(
        scene_name="dense",
        horizon=15,
        num_samples=250,
        temperature=8.0,
        dt=0.2,
        execute_steps=150,
        use_goal_warm_start=False,
        use_anisotropic_sampling=False,
        sdf_influence_distance=1.2,
        sigma_parallel=0.08,
        sigma_perp=0.01,
    )

    print()
    print("=== MuJoCo Main Wrapper Result ===")
    print(result)


if __name__ == "__main__":
    main()
