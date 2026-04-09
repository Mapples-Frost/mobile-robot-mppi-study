import math
import random


def step(state, control, dt):
    """
    单步状态更新
    state = (x, y, theta)
    control = (v, omega)
    """
    x, y, theta = state
    v, omega = control

    x = x + v * math.cos(theta) * dt
    y = y + v * math.sin(theta) * dt
    theta = theta + omega * dt

    return x, y, theta


def rollout(initial_state, control, dt, steps):
    """
    给定一个固定控制，向前滚动多步，生成整条轨迹
    """
    trajectory = [initial_state]
    state = initial_state

    for _ in range(steps):
        state = step(state, control, dt)
        trajectory.append(state)

    return trajectory


def is_collision_point(state, obstacles, robot_radius):
    """
    判断某个状态点是否碰撞
    obstacles: [(ox, oy, obstacle_radius), ...]
    """
    x, y, _ = state

    for ox, oy, obstacle_radius in obstacles:
        distance = math.hypot(x - ox, y - oy)
        if distance <= robot_radius + obstacle_radius:
            return True

    return False


def is_collision_trajectory(trajectory, obstacles, robot_radius):
    """
    判断整条轨迹是否发生碰撞
    """
    for state in trajectory:
        if is_collision_point(state, obstacles, robot_radius):
            return True
    return False


def trajectory_cost(trajectory, goal, obstacles, robot_radius):
    """
    轨迹代价：
    1. 终点离目标越近越好
    2. 碰撞给大惩罚
    3. 离障碍太近增加安全惩罚
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


def sample_controls(
    nominal_control,
    num_samples,#
    v_std,
    omega_std,
    v_min,
    v_max,
    omega_min,
    omega_max,
):
    """
    围绕 nominal_control 做高斯采样
    """
    nominal_v, nominal_omega = nominal_control
    sampled_controls = []

    for _ in range(num_samples):
        v = random.gauss(nominal_v, v_std)
        omega = random.gauss(nominal_omega, omega_std)

        v = max(v_min, min(v, v_max))
        omega = max(omega_min, min(omega, omega_max))

        sampled_controls.append((v, omega))

    return sampled_controls


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
    steps = 25

    nominal_control = (1.0, 0.2)

    num_samples = 30
    v_std = 0.25
    omega_std = 0.25

    v_min, v_max = 0.0, 1.5
    omega_min, omega_max = -1.0, 1.0

    nominal_trajectory = rollout(initial_state, nominal_control, dt, steps)
    nominal_cost = trajectory_cost(nominal_trajectory, goal, obstacles, robot_radius)
    nominal_collision = is_collision_trajectory(
        nominal_trajectory, obstacles, robot_radius
    )

    sampled_controls = sample_controls(
        nominal_control=nominal_control,
        num_samples=num_samples,
        v_std=v_std,
        omega_std=omega_std,
        v_min=v_min,
        v_max=v_max,
        omega_min=omega_min,
        omega_max=omega_max,
    )

    best_control = None
    best_trajectory = None
    best_cost = float("inf")

    for control in sampled_controls:
        trajectory = rollout(initial_state, control, dt, steps)
        cost = trajectory_cost(trajectory, goal, obstacles, robot_radius)

        if cost < best_cost:
            best_cost = cost
            best_control = control
            best_trajectory = trajectory

    best_collision = is_collision_trajectory(best_trajectory, obstacles, robot_radius)
    best_final_state = best_trajectory[-1]

    print("Nominal control:", nominal_control)
    print("Nominal cost:", round(nominal_cost, 3))
    print("Nominal collision:", nominal_collision)
    print()

    print("Best sampled control:", best_control)
    print("Best cost:", round(best_cost, 3))
    print("Best control collision:", best_collision)
    print("Best final state:", tuple(round(v, 3) for v in best_final_state))


if __name__ == "__main__":
    main()