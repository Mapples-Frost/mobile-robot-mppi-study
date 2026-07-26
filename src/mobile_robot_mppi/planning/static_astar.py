"""Deterministic static-only A* used to refresh a soft global reference.

The planner deliberately accepts only frozen static geometry. Dynamic tracks
and forecasts are excluded by contract; they remain the responsibility of
probabilistic MPPI.
"""

from __future__ import annotations

import heapq
import math
from typing import Mapping, Sequence

import numpy as np


def _point_clearance(x, y, obstacles, robot_radius):
    minimum = float("inf")
    for obstacle in obstacles:
        kind = str(obstacle.get("type", "cylinder"))
        if kind == "box":
            center = np.asarray(obstacle["position"][:2], dtype=np.float64)
            half_size = np.asarray(obstacle["size"][:2], dtype=np.float64)
            yaw = float(obstacle.get("yaw", 0.0))
            delta = np.asarray((x, y), dtype=np.float64) - center
            cosine, sine = math.cos(yaw), math.sin(yaw)
            local = np.asarray((
                cosine * delta[0] + sine * delta[1],
                -sine * delta[0] + cosine * delta[1],
            ))
            signed = np.abs(local) - half_size
            outside = float(np.linalg.norm(np.maximum(signed, 0.0)))
            inside = min(max(float(signed[0]), float(signed[1])), 0.0)
            surface = outside + inside
        elif kind == "segment":
            start = np.asarray(obstacle["start"][:2], dtype=np.float64)
            end = np.asarray(obstacle["end"][:2], dtype=np.float64)
            vector = end - start
            denominator = float(np.dot(vector, vector))
            point = np.asarray((x, y), dtype=np.float64)
            if denominator <= 1.0e-12:
                centerline = float(np.linalg.norm(point - start))
            else:
                fraction = float(np.clip(
                    np.dot(point - start, vector) / denominator, 0.0, 1.0
                ))
                centerline = float(
                    np.linalg.norm(point - (start + fraction * vector))
                )
            surface = centerline - float(obstacle.get("thickness", 0.10))
        elif kind == "cylinder":
            center = np.asarray(obstacle["position"][:2], dtype=np.float64)
            surface = (
                float(np.linalg.norm(np.asarray((x, y)) - center))
                - float(obstacle.get("radius", 0.25))
            )
        else:
            raise ValueError(
                "unsupported static A* obstacle type: %s" % kind
            )
        minimum = min(minimum, surface - float(robot_radius))
    return minimum


def _line_is_free(start, end, obstacles, robot_radius, margin, spacing):
    delta = np.asarray(end, dtype=np.float64) - np.asarray(
        start, dtype=np.float64
    )
    distance = float(np.linalg.norm(delta))
    count = max(1, int(math.ceil(distance / float(spacing))))
    return all(
        _point_clearance(
            *(np.asarray(start) + fraction * delta),
            obstacles,
            robot_radius,
        ) >= margin
        for fraction in np.linspace(0.0, 1.0, count + 1)
    )


def _simplify_path(points, obstacles, robot_radius, margin, spacing):
    values = [np.asarray(point, dtype=np.float64) for point in points]
    if len(values) <= 2:
        return np.asarray(values, dtype=np.float64)
    simplified = [values[0]]
    anchor = 0
    while anchor < len(values) - 1:
        following = len(values) - 1
        while following > anchor + 1 and not _line_is_free(
            values[anchor],
            values[following],
            obstacles,
            robot_radius,
            margin,
            spacing,
        ):
            following -= 1
        simplified.append(values[following])
        anchor = following
    return np.asarray(simplified, dtype=np.float64)


def plan_static_astar_path(
    start: Sequence[float],
    goal: Sequence[float],
    obstacles: Sequence[Mapping[str, object]],
    *,
    robot_radius: float,
    clearance_margin: float,
    resolution: float,
    bounds_padding: float = 0.5,
):
    """Return a deterministic simplified A* path over frozen static geometry."""

    start = np.asarray(start, dtype=np.float64).reshape(-1)[:2]
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)[:2]
    obstacles = tuple(obstacles)
    numeric = (
        *start,
        *goal,
        robot_radius,
        clearance_margin,
        resolution,
        bounds_padding,
    )
    if (
        not np.isfinite(numeric).all()
        or robot_radius <= 0.0
        or clearance_margin < 0.0
        or resolution <= 0.0
        or bounds_padding <= 0.0
    ):
        raise ValueError("static A* inputs must be finite and feasible")
    if _point_clearance(*start, obstacles, robot_radius) < 0.0:
        raise ValueError("static A* start lies inside static geometry")
    if _point_clearance(*goal, obstacles, robot_radius) < clearance_margin:
        raise ValueError("static A* goal lacks the requested clearance")

    points_x = [float(start[0]), float(goal[0])]
    points_y = [float(start[1]), float(goal[1])]
    for obstacle in obstacles:
        kind = str(obstacle.get("type", "cylinder"))
        if kind == "segment":
            thickness = float(obstacle.get("thickness", 0.10))
            for point in (obstacle["start"], obstacle["end"]):
                points_x.extend((
                    float(point[0]) - thickness,
                    float(point[0]) + thickness,
                ))
                points_y.extend((
                    float(point[1]) - thickness,
                    float(point[1]) + thickness,
                ))
        elif kind == "box":
            center = np.asarray(
                obstacle["position"][:2], dtype=np.float64
            )
            half_size = np.asarray(
                obstacle["size"][:2], dtype=np.float64
            )
            yaw = float(obstacle.get("yaw", 0.0))
            cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
            axis_aligned_half_size = np.asarray((
                cosine * half_size[0] + sine * half_size[1],
                sine * half_size[0] + cosine * half_size[1],
            ))
            points_x.extend((
                center[0] - axis_aligned_half_size[0],
                center[0] + axis_aligned_half_size[0],
            ))
            points_y.extend((
                center[1] - axis_aligned_half_size[1],
                center[1] + axis_aligned_half_size[1],
            ))
        elif kind == "cylinder":
            center = np.asarray(
                obstacle["position"][:2], dtype=np.float64
            )
            radius = float(obstacle.get("radius", 0.25))
            points_x.extend((center[0] - radius, center[0] + radius))
            points_y.extend((center[1] - radius, center[1] + radius))
        else:
            raise ValueError(
                "unsupported static A* obstacle type: %s" % kind
            )
    padding = (
        float(robot_radius)
        + float(clearance_margin)
        + float(bounds_padding)
    )
    xs = np.arange(
        min(points_x) - padding,
        max(points_x) + padding + resolution,
        resolution,
    )
    ys = np.arange(
        min(points_y) - padding,
        max(points_y) + padding + resolution,
        resolution,
    )
    free = np.asarray([
        [
            _point_clearance(x, y, obstacles, robot_radius)
            >= clearance_margin
            for x in xs
        ]
        for y in ys
    ], dtype=bool)

    start_index = (
        int(np.argmin(np.abs(ys - start[1]))),
        int(np.argmin(np.abs(xs - start[0]))),
    )
    goal_index = (
        int(np.argmin(np.abs(ys - goal[1]))),
        int(np.argmin(np.abs(xs - goal[0]))),
    )
    if not free[start_index]:
        # The continuous robot pose can be safe while the nearest grid centre
        # falls inside the margin. Permit only that start cell.
        free[start_index] = True
    if not free[goal_index]:
        raise ValueError("static A* goal grid cell is occupied")

    moves = (
        (-1, 0), (0, -1), (0, 1), (1, 0),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    distances = {start_index: 0.0}
    parents = {}
    queue = [(0.0, 0.0, start_index)]
    while queue:
        _estimate, distance, current = heapq.heappop(queue)
        if distance != distances.get(current):
            continue
        if current == goal_index:
            break
        for dy, dx in moves:
            following = (current[0] + dy, current[1] + dx)
            if not (
                0 <= following[0] < len(ys)
                and 0 <= following[1] < len(xs)
                and free[following]
            ):
                continue
            if dy and dx and (
                not free[current[0] + dy, current[1]]
                or not free[current[0], current[1] + dx]
            ):
                continue
            candidate = (
                distance + math.hypot(dx, dy) * float(resolution)
            )
            if candidate >= distances.get(following, float("inf")):
                continue
            distances[following] = candidate
            parents[following] = current
            heuristic = math.hypot(
                following[0] - goal_index[0],
                following[1] - goal_index[1],
            ) * float(resolution)
            heapq.heappush(
                queue, (candidate + heuristic, candidate, following)
            )
    if goal_index not in distances:
        raise RuntimeError("no static-only A* route exists")

    grid_path = []
    current = goal_index
    while current != start_index:
        grid_path.append((xs[current[1]], ys[current[0]]))
        current = parents[current]
    grid_path.append((float(start[0]), float(start[1])))
    grid_path.reverse()
    grid_path[-1] = (float(goal[0]), float(goal[1]))
    return _simplify_path(
        grid_path,
        obstacles,
        robot_radius,
        clearance_margin,
        max(0.5 * float(resolution), 0.02),
    )


__all__ = ["plan_static_astar_path"]
