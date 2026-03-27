"""
算法一开始先设定一个 nominal_control，
再用它初始化一条长度为 horizon 的 nominal_sequence。
然后围绕这条参考序列做高斯扰动采样，得到 num_samples 条候选控制序列。
每条候选序列 rollout 成一条轨迹，并分别计算 cost。再根据这些 costs 计算 weights，把
所有候选控制序列加权融合成一条新的 updated_sequence。
接着只执行 updated_sequence 的第一个控制，让真实状态前进一步。
执行完以后，把这条 updated_sequence 往前滚一格，删掉已经执行过的第一个控制，
保留后面的 horizon-1 个控制，并在尾部补一个 tail_control，
得到下一轮的 nominal_sequence。然后重复这个流程。
"""

import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def wrap_angle(theta):
    """
    把角度规范到 [-pi, pi]
    theta = heading angle，朝向角
    """
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def step(state, control, dt):
    """
    单步状态更新

    state = (x, y, theta)
    control = (v, omega)
    dt = 时间步长

    x: x 坐标
    y: y 坐标
    theta: heading angle，朝向角
    v: linear velocity，线速度
    omega: angular velocity，角速度
    """
    x, y, theta = state
    v, omega = control

    next_x = x + v * math.cos(theta) * dt
    next_y = y + v * math.sin(theta) * dt
    next_theta = wrap_angle(theta + omega * dt)

    return next_x, next_y, next_theta


def rollout_control_sequence(start_state, control_sequence, dt):
    """
    从某个起点状态出发，把一整条控制序列 rollout 成一条轨迹

    rollout = 沿着动力学模型向前展开
    control_sequence = [(v1, omega1), (v2, omega2), ...]
    trajectory = [(x0, y0, theta0), (x1, y1, theta1), ...]
    """
    trajectory = [start_state]
    current_state = start_state

    for control in control_sequence:# sequence里面的每一个control是作为每一步的参数参与进来的
        current_state = step(current_state, control, dt)
        trajectory.append(current_state)# rollout中最后的每一条轨迹对应多少条states取决于horizon

    return trajectory #返回的是某一连续序列每一步不同的control跑出来的trajctory


def initialize_control_sequence(nominal_control, horizon):
    """
    生成初始参考控制序列

    nominal = 名义上的、参考的
    horizon = 预测时域长度
    """
    return [nominal_control for _ in range(horizon)]


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
    围绕 nominal_sequence 采样出很多条候选控制序列

    num_samples = 候选序列数量
    v_std = 线速度采样标准差
    omega_std = 角速度采样标准差
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

    if len(sampled_sequences) > 0:
        sampled_sequences[0] = nominal_sequence.copy()

    return sampled_sequences


def obstacle_clearance(x, y, obstacle, robot_radius):
    """
    计算机器人当前位置到某个圆形障碍物边界的净空距离

    obstacle = (obs_x, obs_y, obs_radius)
    clearance = 净空距离
    """
    obs_x, obs_y, obs_radius = obstacle
    center_distance = math.hypot(x - obs_x, y - obs_y)
    clearance = center_distance - (robot_radius + obs_radius)
    return clearance


def trajectory_cost(trajectory, control_sequence, goal, obstacles, robot_radius):
    """
    计算一条轨迹的总代价

    goal = (goal_x, goal_y)
    返回:
        total_cost: 总代价
        collided: 是否碰撞
    """
    goal_x, goal_y = goal
    total_cost = 0.0
    collided = False

    for i, state in enumerate(trajectory[1:], start=1):
        x, y, theta = state

        '''
        第一类代价：距离目标的距离，这里在罚：每一步都不要离目标太远
        '''
        goal_distance = math.hypot(goal_x - x, goal_y - y)
        total_cost += 0.15 * goal_distance
        '''
        第二类代价：朝向误差，这里在罚：每一步的朝向和目标方向不要差太远
    
        '''

        desired_heading = math.atan2(goal_y - y, goal_x - x)
        heading_error = abs(wrap_angle(desired_heading - theta))
        total_cost += 0.05 * heading_error
        '''
        第三类代价：角速度自身惩罚，这里在罚：不要转的越猛
        '''

        v, omega = control_sequence[i - 1]
        total_cost += 0.01 * (omega ** 2)
        '''
        第四类代价：障碍物代价，这里在罚：不要碰到障碍物
        '''

        for obstacle in obstacles:
            clearance = obstacle_clearance(x, y, obstacle, robot_radius)

            if clearance <= 0.0:
                collided = True
                total_cost += 10000.0
            elif clearance < 0.8:
                total_cost += 30.0 * (0.8 - clearance) ** 2

    '''
    四五类代价：终点距离代价：最后一步不能离终点太远
    '''
    final_x, final_y, final_theta = trajectory[-1]
    final_goal_distance = math.hypot(goal_x - final_x, goal_y - final_y)
    total_cost += 100.0 * final_goal_distance
    '''
    第六类代价：终点朝向代价：最后一步的朝向不能偏离目标朝向太远
    '''


    final_desired_heading = math.atan2(goal_y - final_y, goal_x - final_x)
    final_heading_error = abs(wrap_angle(final_desired_heading - final_theta))
    total_cost += 2.0 * final_heading_error
    '''
    控制平滑代价：这一步在罚：不要控制变化过猛
    '''

    for i in range(1, len(control_sequence)):
        prev_v, prev_omega = control_sequence[i - 1]
        curr_v, curr_omega = control_sequence[i]
        total_cost += 0.08 * ((curr_v - prev_v) ** 2 + 0.2 * (curr_omega - prev_omega) ** 2)

    return total_cost, collided


def compute_weights(costs, temperature):
    """
    根据每条候选轨迹的代价计算权重

    temperature = 温度参数/平滑参数
    代价越小，权重越大
    """
    min_cost = min(costs)
    shifted_costs = [cost - min_cost for cost in costs]

    unnormalized_weights = [
        math.exp(-shifted_cost / max(temperature, 1e-8))
        for shifted_cost in shifted_costs
    ]

    weight_sum = sum(unnormalized_weights)

    if weight_sum < 1e-12:
        return [1.0 / len(costs)] * len(costs)

    weights = [w / weight_sum for w in unnormalized_weights]
    return weights


def weighted_update_sequence(sampled_sequences, weights):
    """
    按权重把很多条候选控制序列加权平均，得到新的参考控制序列

    weighted = 带权重的
    update = 更新
    """
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
    """
    把控制序列整体向前滚一格

    shift = 平移、前移
    tail_control = 在尾部补上的控制
    """
    if len(sequence) == 0:
        return []

    shifted = sequence[1:].copy()
    shifted.append(tail_control)
    return shifted


def executed_trajectory_cost(executed_trajectory, executed_controls, goal, obstacles, robot_radius):
    """
    单独给“真实执行出来的轨迹”算总代价
    """
    return trajectory_cost(executed_trajectory, executed_controls, goal, obstacles, robot_radius)


def plot_result(executed_trajectory, sampled_best_trajectory, start_state, goal, obstacles, robot_radius):
    """
    画图展示最终结果
    """
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

    ax.set_title("MPPI Receding Horizon Demo")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    plt.show()


def main():
    random.seed(11)
    '''
    参数区：调整各种参数
    '''

    dt = 0.2
    horizon = 25
    num_samples = 250
    execute_steps = 60
    temperature = 8.0

    v_std = 0.25
    omega_std = 0.35

    v_min = 0.0
    v_max = 1.4
    omega_max = 1.2

    goal_tolerance = 0.35
    robot_radius = 0.25

    start_state = (0.0, 0.0, 0.0)
    goal = (5.5, 1.8)

    obstacles = [
        (2.2, 0.7, 0.6),
        (3.4, 1.5, 0.5),
        (4.8, 0.8, 0.5),
    ]

    nominal_control = (1.0, 0.0)
    nominal_sequence = initialize_control_sequence(nominal_control, horizon)

    current_state = start_state
    executed_trajectory = [current_state]
    executed_controls = []

    sampled_best_trajectory = None

    print("=== MPPI Receding Horizon Demo ===")
    print("start_state =", start_state)
    print("goal =", goal)
    print("horizon =", horizon)
    print("num_samples =", num_samples)
    print("temperature =", temperature)
    print()
    '''
    主循环区，进行主体大循环
    '''
    for step_idx in range(execute_steps):
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
            )

            sampled_trajectories.append(trajectory)
            costs.append(total_cost)
            collisions.append(collided)

        weights = compute_weights(costs, temperature)
        updated_sequence = weighted_update_sequence(sampled_sequences, weights)

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
            print(f"Reached goal tolerance at step {step_idx:02d}.")
            break

    final_cost, final_collision = executed_trajectory_cost(
        executed_trajectory=executed_trajectory,
        executed_controls=executed_controls,
        goal=goal,
        obstacles=obstacles,
        robot_radius=robot_radius,
    )

    final_state = executed_trajectory[-1]

    print()
    print("=== Final Result ===")
    print("Executed trajectory length =", len(executed_trajectory))
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
    )


if __name__ == "__main__":
    main()