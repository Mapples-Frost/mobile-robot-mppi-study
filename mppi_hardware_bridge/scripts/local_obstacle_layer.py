# -*- coding: utf-8 -*-
from __future__ import print_function

import math


def is_finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return not (math.isnan(value) or math.isinf(value))


def scan_to_local_points(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    max_radius,
    min_radius=0.10,
    angle_offset_rad=0.0,
    downsample_step=3,
):
    """
    Convert raw LaserScan fields into local obstacle points.

    Convention:
      local angle 0 is robot front (+x)
      x_local = r * cos(local_angle)
      y_local = r * sin(local_angle)

    angle_offset_rad is the LaserScan angle that corresponds to robot front.
    """
    try:
        step = int(downsample_step)
    except (TypeError, ValueError):
        step = 1
    if step <= 0:
        step = 1

    angle_min = float(angle_min)
    angle_increment = float(angle_increment)
    range_min = float(range_min)
    range_max = float(range_max)
    max_radius = float(max_radius)
    min_radius = float(min_radius)
    angle_offset_rad = float(angle_offset_rad)

    local_points = []

    for i, raw_range in enumerate(ranges):
        if i % step != 0:
            continue

        if not is_finite_number(raw_range):
            continue

        r = float(raw_range)

        if r < range_min or r > range_max:
            continue
        if r < min_radius or r > max_radius:
            continue

        scan_angle = angle_min + float(i) * angle_increment
        local_angle = scan_angle - angle_offset_rad

        x_local = r * math.cos(local_angle)
        y_local = r * math.sin(local_angle)
        local_points.append((x_local, y_local))

    return local_points


def local_points_to_circular_obstacles(local_points, obstacle_radius=0.08):
    obstacle_radius = float(obstacle_radius)
    obstacles = []

    for point in local_points:
        x, y = point[0], point[1]
        obstacles.append((float(x), float(y), obstacle_radius))

    return obstacles


def local_points_to_experiment_points(local_points, current_state_exp):
    x_robot = float(current_state_exp[0])
    y_robot = float(current_state_exp[1])
    yaw_robot = float(current_state_exp[2])

    cos_yaw = math.cos(yaw_robot)
    sin_yaw = math.sin(yaw_robot)

    experiment_points = []

    for point in local_points:
        x_local = float(point[0])
        y_local = float(point[1])

        x_obs_exp = x_robot + cos_yaw * x_local - sin_yaw * y_local
        y_obs_exp = y_robot + sin_yaw * x_local + cos_yaw * y_local
        experiment_points.append((x_obs_exp, y_obs_exp))

    return experiment_points


def scan_to_experiment_obstacles(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    current_state_exp,
    max_radius,
    min_radius=0.10,
    angle_offset_rad=0.0,
    downsample_step=3,
    obstacle_radius=0.08,
):
    local_points = scan_to_local_points(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        max_radius=max_radius,
        min_radius=min_radius,
        angle_offset_rad=angle_offset_rad,
        downsample_step=downsample_step,
    )

    experiment_points = local_points_to_experiment_points(
        local_points=local_points,
        current_state_exp=current_state_exp,
    )

    return local_points_to_circular_obstacles(
        experiment_points,
        obstacle_radius=obstacle_radius,
    )


def _run_basic_tests():
    print("local_obstacle_layer basic tests")
    print("--------------------------------")

    points = scan_to_local_points(
        ranges=[1.0],
        angle_min=0.0,
        angle_increment=0.0,
        range_min=0.05,
        range_max=6.0,
        max_radius=2.0,
        downsample_step=1,
    )
    print("front 1m point:", points)
    assert len(points) == 1
    assert abs(points[0][0] - 1.0) < 1e-6
    assert abs(points[0][1]) < 1e-6

    points = scan_to_local_points(
        ranges=[1.0],
        angle_min=math.pi / 4.0,
        angle_increment=0.0,
        range_min=0.05,
        range_max=6.0,
        max_radius=2.0,
        downsample_step=1,
    )
    print("left-front point:", points)
    assert len(points) == 1
    assert points[0][1] > 0.0

    points = scan_to_local_points(
        ranges=[float("inf"), float("nan"), 5.0, 0.05, 0.5],
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.10,
        range_max=6.0,
        max_radius=2.0,
        min_radius=0.10,
        downsample_step=1,
    )
    print("filtered points:", points)
    assert len(points) == 1

    points = scan_to_local_points(
        ranges=[1.0, 1.0, 1.0, 1.0, 1.0],
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.05,
        range_max=6.0,
        max_radius=2.0,
        downsample_step=2,
    )
    print("downsampled points:", points)
    assert len(points) == 3

    obstacles = local_points_to_circular_obstacles(points, obstacle_radius=0.08)
    print("circular obstacles:", obstacles)
    assert len(obstacles) == len(points)
    assert abs(obstacles[0][2] - 0.08) < 1e-9

    exp_points = local_points_to_experiment_points(
        local_points=[(1.0, 0.0)],
        current_state_exp=(0.0, 0.0, 0.0),
    )
    print("experiment point at origin:", exp_points)
    assert len(exp_points) == 1
    assert abs(exp_points[0][0] - 1.0) < 1e-6
    assert abs(exp_points[0][1]) < 1e-6

    exp_points = local_points_to_experiment_points(
        local_points=[(1.0, 0.0)],
        current_state_exp=(1.0, 2.0, 0.0),
    )
    print("translated experiment point:", exp_points)
    assert len(exp_points) == 1
    assert abs(exp_points[0][0] - 2.0) < 1e-6
    assert abs(exp_points[0][1] - 2.0) < 1e-6

    exp_points = local_points_to_experiment_points(
        local_points=[(1.0, 0.0)],
        current_state_exp=(0.0, 0.0, math.pi / 2.0),
    )
    print("rotated experiment point:", exp_points)
    assert len(exp_points) == 1
    assert abs(exp_points[0][0]) < 1e-6
    assert abs(exp_points[0][1] - 1.0) < 1e-6

    obstacles = scan_to_experiment_obstacles(
        ranges=[1.0],
        angle_min=0.0,
        angle_increment=0.0,
        range_min=0.05,
        range_max=6.0,
        current_state_exp=(1.0, 2.0, 0.0),
        max_radius=2.0,
        downsample_step=1,
        obstacle_radius=0.08,
    )
    print("experiment obstacles:", obstacles)
    assert len(obstacles) == 1
    assert abs(obstacles[0][0] - 2.0) < 1e-6
    assert abs(obstacles[0][1] - 2.0) < 1e-6
    assert abs(obstacles[0][2] - 0.08) < 1e-9

    print("")
    print("All local_obstacle_layer basic tests passed.")


if __name__ == "__main__":
    _run_basic_tests()
