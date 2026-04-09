import math
import random


def step(state, control, dt):
    """
    step: 单步更新
    state: 状态，格式是 (x, y, theta)
    control: 控制输入，格式是 (v, omega)
    dt: 时间步长
    """
    x, y, theta = state
    v, omega = control

    x = x + v * math.cos(theta) * dt
    y = y + v * math.sin(theta) * dt
    theta = theta + omega * dt

    return x, y, theta


def rollout_control_sequence(initial_state, control_sequence, dt):
    """
    rollout_control_sequence:
    rollout = 向前滚动展开
    control_sequence = 控制序列，一串未来控制输入

    作用：
    给定初始状态和一整串控制，依次往前推，
    最后得到整条未来轨迹。
    """
    trajectory = [initial_state]
    state = initial_state

    for control in control_sequence:
        state = step(state, control, dt)
        trajectory.append(state)

    return trajectory


def is_collision_point(state, obstacles, robot_radius):
    """
    is_collision_point:
    is = 是否
    collision = 碰撞
    point = 单个点

    作用：
    判断某一个状态点是否撞到障碍物。
    """
    x, y, _ = state

    for ox, oy, obstacle_radius in obstacles:
        distance = math.hypot(x - ox, y - oy)
        if distance <= robot_radius + obstacle_radius:
            return True

    return False


def is_collision_trajectory(trajectory, obstacles, robot_radius):
    """
    is_collision_trajectory:
    trajectory = 轨迹

    作用：
    判断整条轨迹是否发生碰撞。
    """
    for state in trajectory:
        if is_collision_point(state, obstacles, robot_radius):
            return True
    return False


def trajectory_cost(trajectory, goal, obstacles, robot_radius):
    """
    trajectory_cost:
    trajectory = 轨迹
    cost = 代价 / 评分

    作用：
    给一条轨迹打分，代价越小越好。
    当前包含：
    1. 终点到目标点的距离
    2. 碰撞大惩罚
    3. 离障碍太近的安全惩罚
    """
    end_x, end_y, _ = trajectory[-1]
    goal_x, goal_y = goal

    goal_distance_cost = math.hypot(end_x - goal_x, end_y - goal_y)

    collision_penalty = 0.0
    if is_collision_trajectory(trajectory, obstacles, robot_radius):
        collision_penalty = 1000.0

    min_clearance = float("inf")
    for x, y, _ in trajectory:
        for ox, oy, obstacle_radius in obstacles:
            clearance = math.hypot(x - ox, y - oy) - (robot_radius + obstacle_radius)
            if clearance < min_clearance:
                min_clearance = clearance

    safety_penalty = 0.0
    safe_distance = 0.5
    if min_clearance < safe_distance:
        safety_penalty = (safe_distance - min_clearance) * 10.0

    return goal_distance_cost + collision_penalty + safety_penalty


def initialize_control_sequence(horizon, nominal_control):
    """
    initialize_control_sequence:
    initialize = 初始化
    control_sequence = 控制序列
    horizon = 预测时域 / 往前看多少步
    nominal_control = 名义控制 / 参考控制 / 中心控制

    作用：
    先生成一条“初始控制序列”。
    最简单的办法就是：
    用同一个 nominal_control 重复 horizon 次。
    """
    return [nominal_control for _ in range(horizon)]


def sample_control_sequences(
    nominal_sequence,
    num_samples,
    v_std,
    omega_std,
    v_min,
    v_max,
    omega_min,
    omega_max,
):
    """
    sample_control_sequences:
    sample = 采样
    control_sequences = 多组控制序列

    nominal_sequence = 当前参考控制序列
    num_samples = 采样多少组候选序列
    v_std / omega_std = 高斯噪声标准差

    作用：
    围绕 nominal_sequence 的每一个时间步控制，
    加入随机扰动，生成很多组候选控制序列。
    """
    sampled_sequences = []

    for _ in range(num_samples):
        candidate_sequence = []

        for nominal_control in nominal_sequence:
            nominal_v, nominal_omega = nominal_control

            v = random.gauss(nominal_v, v_std)
            omega = random.gauss(nominal_omega, omega_std)

            v = max(v_min, min(v, v_max))
            omega = max(omega_min, min(omega, omega_max))

            candidate_sequence.append((v, omega))

        sampled_sequences.append(candidate_sequence)

    return sampled_sequences


def main():
    random.seed(42)

    initial_state = (1.0, 1.0, 0.0)
    goal = (8.0, 8.0)

    obstacles = [
        (4.0, 4.0, 1.0),
        (5.5, 6.0, 0.8),
        (6.0, 3.5, 0.7),
    ]

    robot_radius = 0.3
    dt = 0.1

    horizon = 15
    nominal_control = (1.0, 0.2)

    num_samples = 40
    v_std = 0.20
    omega_std = 0.20

    v_min, v_max = 0.0, 1.5
    omega_min, omega_max = -1.0, 1.0

    nominal_sequence = initialize_control_sequence(horizon, nominal_control)
    nominal_trajectory = rollout_control_sequence(initial_state, nominal_sequence, dt)
    nominal_cost = trajectory_cost(
        nominal_trajectory, goal, obstacles, robot_radius
    )
    nominal_collision = is_collision_trajectory(
        nominal_trajectory, obstacles, robot_radius
    )

    sampled_sequences = sample_control_sequences(
        nominal_sequence=nominal_sequence,
        num_samples=num_samples,
        v_std=v_std,
        omega_std=omega_std,
        v_min=v_min,
        v_max=v_max,
        omega_min=omega_min,
        omega_max=omega_max,
    )

    best_sequence = None
    best_trajectory = None
    best_cost = float("inf")

    for control_sequence in sampled_sequences:
        trajectory = rollout_control_sequence(initial_state, control_sequence, dt)
        cost = trajectory_cost(trajectory, goal, obstacles, robot_radius)

        if cost < best_cost:
            best_cost = cost
            best_sequence = control_sequence
            best_trajectory = trajectory

    best_collision = is_collision_trajectory(
        best_trajectory, obstacles, robot_radius
    )
    best_final_state = best_trajectory[-1]

    print("Nominal sequence length:", len(nominal_sequence))
    print("Nominal first 3 controls:", nominal_sequence[:3])
    print("Nominal cost:", round(nominal_cost, 3))
    print("Nominal collision:", nominal_collision)
    print()

    print("Number of sampled sequences:", len(sampled_sequences))
    print("Best cost:", round(best_cost, 3))
    print("Best collision:", best_collision)
    print("Best first 3 controls:", best_sequence[:3])
    print("Best final state:", tuple(round(v, 3) for v in best_final_state))


if __name__ == "__main__":
    main()