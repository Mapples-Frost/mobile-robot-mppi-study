# -*- coding: utf-8 -*-
from __future__ import print_function

import math


def _point_xy(point):
    return float(point[0]), float(point[1])


def distance_to_nearest_obstacle_point(x, y, obstacle_points):
    if not obstacle_points:
        return float("inf")

    x = float(x)
    y = float(y)
    best_distance = float("inf")

    for point in obstacle_points:
        obs_x, obs_y = _point_xy(point)
        distance = math.hypot(x - obs_x, y - obs_y)
        if distance < best_distance:
            best_distance = distance

    return best_distance


def local_obstacle_penalty_for_state(
    x,
    y,
    obstacle_points,
    influence_distance,
    weight,
    hard_collision_distance=0.15,
    hard_collision_cost=20000.0,
):
    if not obstacle_points:
        return 0.0

    d_min = distance_to_nearest_obstacle_point(x, y, obstacle_points)
    cost = 0.0

    if d_min <= float(hard_collision_distance):
        cost += float(hard_collision_cost)

    influence_distance = float(influence_distance)
    if d_min < influence_distance:
        cost += float(weight) * (influence_distance - d_min) ** 2

    return cost


def local_obstacle_penalty_for_trajectory(
    trajectory,
    obstacle_points,
    influence_distance,
    weight,
    hard_collision_distance=0.15,
    hard_collision_cost=20000.0,
):
    if not trajectory or not obstacle_points:
        return 0.0

    total_cost = 0.0

    for state in trajectory:
        x = state[0]
        y = state[1]
        total_cost += local_obstacle_penalty_for_state(
            x=x,
            y=y,
            obstacle_points=obstacle_points,
            influence_distance=influence_distance,
            weight=weight,
            hard_collision_distance=hard_collision_distance,
            hard_collision_cost=hard_collision_cost,
        )

    return total_cost


def _run_basic_tests():
    print("local_obstacle_cost basic tests")
    print("-------------------------------")

    cost = local_obstacle_penalty_for_state(
        x=0.0,
        y=0.0,
        obstacle_points=[],
        influence_distance=0.7,
        weight=20.0,
    )
    print("no obstacle cost:", cost)
    assert cost == 0.0

    cost = local_obstacle_penalty_for_state(
        x=0.0,
        y=0.0,
        obstacle_points=[(10.0, 0.0)],
        influence_distance=0.7,
        weight=20.0,
    )
    print("far obstacle cost:", cost)
    assert cost == 0.0

    near_cost = local_obstacle_penalty_for_state(
        x=0.0,
        y=0.0,
        obstacle_points=[(0.30, 0.0)],
        influence_distance=0.7,
        weight=20.0,
    )
    print("near obstacle cost:", near_cost)
    assert near_cost > 0.0

    collision_cost = local_obstacle_penalty_for_state(
        x=0.0,
        y=0.0,
        obstacle_points=[(0.05, 0.0)],
        influence_distance=0.7,
        weight=20.0,
    )
    print("collision cost:", collision_cost)
    assert collision_cost > 10000.0

    trajectory_cost = local_obstacle_penalty_for_trajectory(
        trajectory=[(0.0, 0.0, 0.0), (0.20, 0.0, 0.0), (0.40, 0.0, 0.0)],
        obstacle_points=[(0.30, 0.0)],
        influence_distance=0.7,
        weight=20.0,
    )
    print("trajectory cost:", trajectory_cost)
    assert trajectory_cost > near_cost

    print("")
    print("All local_obstacle_cost basic tests passed.")


if __name__ == "__main__":
    _run_basic_tests()
