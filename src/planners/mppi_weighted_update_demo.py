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
    control_sequence = 控制序列

    作用：
    给一整条控制序列做 forward simulation（前向模拟），
    得到整条未来轨迹。
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
    判断某一个状态点是否发生碰撞。
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
    cost = 代价

    作用：
    给轨迹打分，代价越小越好。
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
    nominal_control = 名义控制 / 参考控制

    作用：
    用同一个 nominal_control 重复 horizon 次，
    生成初始参考控制序列。
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
    control_sequences = 多条控制序列

    作用：
    围绕 nominal_sequence 的每一个时间步控制做高斯采样，
    生成很多条候选控制序列。
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


def compute_weights(costs, temperature):
    """
    compute_weights:
    compute = 计算
    weights = 权重
    costs = 各条轨迹的代价
    temperature = 温度参数 / 平滑参数
    temperature决定了cost的相对差距

    作用：
    把一堆 cost 转成一堆 weight。
    cost 越小，对应的 weight 越大。

    这里用了指数形式：
    weight ~ exp(-cost / temperature)

    为了数值更稳定，先减去最小 cost。
    """
    min_cost = min(costs)

    shifted_costs = []
    for cost in costs:
        shifted_costs.append(cost - min_cost)

    exp_values = []
    for shifted_cost in shifted_costs:
        value = math.exp(-shifted_cost / temperature)
        exp_values.append(value)

    weight_sum = sum(exp_values)

    weights = []
    for value in exp_values:
        weights.append(value / weight_sum)

    return weights


def weighted_update_sequence(sampled_sequences, weights):
    """
    weighted_update_sequence:
    weighted = 带权重的
    update = 更新
    sequence = 序列

    作用：
    不再只选最优的一条候选序列，
    而是把所有候选序列按权重做加权平均，
    得到新的控制序列。
    """
    horizon = len(sampled_sequences[0])
    num_samples = len(sampled_sequences)

    updated_sequence = []

    for t in range(horizon):
        weighted_v = 0.0
        weighted_omega = 0.0

        for k in range(num_samples):
            v, omega = sampled_sequences[k][t]
            weighted_v += weights[k] * v
            weighted_omega += weights[k] * omega

        updated_sequence.append((weighted_v, weighted_omega))

    return updated_sequence


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

    num_samples = 60
    v_std = 0.20
    omega_std = 0.20

    v_min, v_max = 0.0, 1.5
    omega_min, omega_max = -1.0, 1.0

    temperature = 0.5

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

    sampled_costs = []
    sampled_trajectories = []

    for control_sequence in sampled_sequences:
        trajectory = rollout_control_sequence(initial_state, control_sequence, dt)
        cost = trajectory_cost(trajectory, goal, obstacles, robot_radius)

        sampled_trajectories.append(trajectory)
        sampled_costs.append(cost)

    weights = compute_weights(sampled_costs, temperature)
    updated_sequence = weighted_update_sequence(sampled_sequences, weights)

    updated_trajectory = rollout_control_sequence(initial_state, updated_sequence, dt)
    updated_cost = trajectory_cost(
        updated_trajectory, goal, obstacles, robot_radius
    )
    updated_collision = is_collision_trajectory(
        updated_trajectory, obstacles, robot_radius
    )

    best_index = min(range(len(sampled_costs)), key=lambda i: sampled_costs[i])
    best_sampled_sequence = sampled_sequences[best_index]
    best_sampled_cost = sampled_costs[best_index]

    print("Nominal cost:", round(nominal_cost, 3))
    print("Nominal collision:", nominal_collision)
    print("Nominal first 3 controls:", nominal_sequence[:3])
    print()

    print("Best sampled cost:", round(best_sampled_cost, 3))
    print("Best sampled first 3 controls:", best_sampled_sequence[:3])
    print()

    print("Updated cost:", round(updated_cost, 3))
    print("Updated collision:", updated_collision)
    print("Updated first 3 controls:", updated_sequence[:3])
    print("Weight sum:", round(sum(weights), 6))
    print("Top 5 weights:", [round(w, 4) for w in sorted(weights, reverse=True)[:5]])


if __name__ == "__main__":
    main()