"""Offline geometry checks for static scenes; never used as planner input."""

import heapq
import math

import numpy as np


def point_clearance(x, y, obstacles, robot_radius):
    values = []
    for obstacle in obstacles:
        px, py = (float(v) for v in obstacle.get("position", (0.0, 0.0)))
        if str(obstacle.get("type", "cylinder")) == "box":
            sx, sy = (float(v) for v in obstacle.get("size", (0.25, 0.25)))
            yaw = float(obstacle.get("yaw", 0.0))
            dx, dy = float(x) - px, float(y) - py
            local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
            local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
            outside_x = max(abs(local_x) - sx, 0.0)
            outside_y = max(abs(local_y) - sy, 0.0)
            outside = math.hypot(outside_x, outside_y)
            inside = min(max(abs(local_x) - sx, abs(local_y) - sy), 0.0)
            values.append(outside + inside - float(robot_radius))
        else:
            values.append(
                math.hypot(float(x) - px, float(y) - py)
                - float(obstacle.get("radius", 0.25))
                - float(robot_radius)
            )
    return min(values) if values else float("inf")


def audit_static_scene(
    scene, start, goal, robot_radius, margin=0.03, resolution=0.05,
    include_path=False,
):
    if resolution <= 0.0 or margin < 0.0 or robot_radius <= 0.0:
        raise ValueError("resolution and robot radius must be positive; margin non-negative")
    obstacles = tuple(scene.get("obstacles", ()))
    points_x = [float(start[0]), float(goal[0])]
    points_y = [float(start[1]), float(goal[1])]
    for obstacle in obstacles:
        px, py = obstacle.get("position", (0.0, 0.0))
        extent = max(
            tuple(float(v) for v in obstacle.get("size", ()))
            + (float(obstacle.get("radius", 0.0)),)
        )
        points_x.extend((float(px) - extent, float(px) + extent))
        points_y.extend((float(py) - extent, float(py) + extent))
    padding = float(robot_radius) + float(margin) + 0.5
    xs = np.arange(min(points_x) - padding, max(points_x) + padding + resolution, resolution)
    ys = np.arange(min(points_y) - padding, max(points_y) + padding + resolution, resolution)
    free = np.asarray([
        [point_clearance(x, y, obstacles, robot_radius) >= margin for x in xs]
        for y in ys
    ], dtype=bool)
    start_index = (int(np.argmin(np.abs(ys - start[1]))), int(np.argmin(np.abs(xs - start[0]))))
    goal_index = (int(np.argmin(np.abs(ys - goal[1]))), int(np.argmin(np.abs(xs - goal[0]))))
    distances = {start_index: 0.0}
    parents = {}
    queue = [(0.0, start_index)]
    if free[start_index] and free[goal_index]:
        while queue:
            distance, current = heapq.heappop(queue)
            if current == goal_index:
                break
            if distance != distances[current]:
                continue
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                following = (current[0] + dy, current[1] + dx)
                if not (0 <= following[0] < len(ys) and 0 <= following[1] < len(xs)):
                    continue
                if not free[following]:
                    continue
                candidate = distance + math.hypot(dx, dy) * float(resolution)
                if candidate < distances.get(following, float("inf")):
                    distances[following] = candidate
                    parents[following] = current
                    heapq.heappush(queue, (candidate, following))
    start_clearance = point_clearance(start[0], start[1], obstacles, robot_radius)
    goal_clearance = point_clearance(goal[0], goal[1], obstacles, robot_radius)
    shortest = distances.get(goal_index)
    report = {
        "scene": str(scene.get("name", "unknown")),
        "start_clearance": float(start_clearance),
        "goal_clearance": float(goal_clearance),
        "start_free": bool(free[start_index]),
        "goal_free": bool(free[goal_index]),
        "path_exists": shortest is not None,
        "grid_shortest_path_length": None if shortest is None else float(shortest),
        "resolution": float(resolution),
        "margin": float(margin),
    }
    if include_path:
        path = []
        if shortest is not None:
            current = goal_index
            while current != start_index:
                path.append((float(xs[current[1]]), float(ys[current[0]])))
                current = parents[current]
            path.append((float(xs[start_index[1]]), float(ys[start_index[0]])))
            path.reverse()
        report["path"] = path
    return report
