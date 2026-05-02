# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import time


EPSILON = 1e-9


def is_finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return not (math.isnan(value) or math.isinf(value))


def normalize_angle(angle):
    angle = float(angle)
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def _coerce_step(value, default_value=1):
    try:
        step = int(value)
    except (TypeError, ValueError):
        step = int(default_value)
    if step <= 0:
        step = int(default_value)
    if step <= 0:
        step = 1
    return step


def _point_xy(point):
    return (float(point["x"]), float(point["y"]))


def _distance_xy(a_x, a_y, b_x, b_y):
    return math.hypot(float(a_x) - float(b_x), float(a_y) - float(b_y))


def _distance_points(a, b):
    return _distance_xy(a["x"], a["y"], b["x"], b["y"])


def _mean_std(values):
    if not values:
        return (0.0, 0.0)

    mean_value = sum(values) / float(len(values))
    variance = 0.0
    for value in values:
        diff = float(value) - mean_value
        variance += diff * diff
    variance /= float(len(values))
    return (mean_value, math.sqrt(max(0.0, variance)))


def _copy_feature_with_edge(feature, is_edge):
    copied = dict(feature)
    copied["is_edge"] = bool(is_edge)
    return copied


def _sample_evenly(items, max_count):
    try:
        max_count = int(max_count)
    except (TypeError, ValueError):
        max_count = len(items)

    if max_count <= 0:
        return []
    if len(items) <= max_count:
        return list(items)
    if max_count == 1:
        return [items[len(items) // 2]]

    sampled = []
    last_index = len(items) - 1
    for sample_index in range(max_count):
        ratio = float(sample_index) / float(max_count - 1)
        item_index = int(round(ratio * float(last_index)))
        sampled.append(items[item_index])
    return sampled


def _nearest_points(points, max_count):
    point_list = list(points)
    try:
        max_count = int(max_count)
    except (TypeError, ValueError):
        max_count = len(point_list)

    if max_count <= 0:
        return []

    def point_range(point):
        if "range" in point:
            return float(point["range"])
        return math.hypot(float(point["x"]), float(point["y"]))

    point_list.sort(key=point_range)
    return point_list[:max_count]


def limit_obstacles_evenly(obstacles, max_count):
    """
    Limit obstacle count without biasing toward the first scan angles.
    """
    obstacle_list = list(obstacles)
    try:
        max_count = int(max_count)
    except (TypeError, ValueError):
        max_count = 0

    if max_count <= 0:
        return list(obstacle_list)
    if len(obstacle_list) <= max_count:
        return list(obstacle_list)

    return _sample_evenly(obstacle_list, max_count)


def _solve_3x3(matrix, vector):
    augmented = []
    for row_index in range(3):
        row = [
            float(matrix[row_index][0]),
            float(matrix[row_index][1]),
            float(matrix[row_index][2]),
            float(vector[row_index]),
        ]
        augmented.append(row)

    for col in range(3):
        pivot_row = col
        pivot_value = abs(augmented[col][col])
        for row_index in range(col + 1, 3):
            candidate = abs(augmented[row_index][col])
            if candidate > pivot_value:
                pivot_value = candidate
                pivot_row = row_index

        if pivot_value <= EPSILON:
            return None

        if pivot_row != col:
            tmp = augmented[col]
            augmented[col] = augmented[pivot_row]
            augmented[pivot_row] = tmp

        pivot = augmented[col][col]
        for value_index in range(col, 4):
            augmented[col][value_index] /= pivot

        for row_index in range(3):
            if row_index == col:
                continue
            factor = augmented[row_index][col]
            if abs(factor) <= EPSILON:
                continue
            for value_index in range(col, 4):
                augmented[row_index][value_index] -= factor * augmented[col][value_index]

    return [augmented[0][3], augmented[1][3], augmented[2][3]]


def _angle_coverage(angles):
    if len(angles) <= 1:
        return 0.0

    wrapped = []
    for angle in angles:
        value = normalize_angle(angle)
        if value < 0.0:
            value += 2.0 * math.pi
        wrapped.append(value)
    wrapped.sort()

    largest_gap = 0.0
    for index in range(1, len(wrapped)):
        gap = wrapped[index] - wrapped[index - 1]
        if gap > largest_gap:
            largest_gap = gap

    end_gap = wrapped[0] + 2.0 * math.pi - wrapped[-1]
    if end_gap > largest_gap:
        largest_gap = end_gap

    coverage = 2.0 * math.pi - largest_gap
    return max(0.0, min(2.0 * math.pi, coverage))


def smooth_ranges(
    ranges,
    window_size=3,
    preserve_invalid=True,
):
    """
    Smooth finite LaserScan ranges with a small moving average window.

    inf/nan values remain invalid. They are not filled from neighbors because
    doing so would invent obstacle returns where the lidar reported no return.
    """
    original = list(ranges)
    try:
        window_size = int(window_size)
    except (TypeError, ValueError):
        window_size = 1

    if window_size <= 1:
        return list(original)

    left_count = (window_size - 1) // 2
    right_count = window_size - 1 - left_count
    smoothed = []

    for index, raw_range in enumerate(original):
        if not is_finite_number(raw_range):
            smoothed.append(raw_range)
            continue

        start = max(0, index - left_count)
        end = min(len(original), index + right_count + 1)
        values = []
        for neighbor_index in range(start, end):
            neighbor_range = original[neighbor_index]
            if is_finite_number(neighbor_range):
                values.append(float(neighbor_range))

        if values:
            smoothed.append(sum(values) / float(len(values)))
        elif preserve_invalid:
            smoothed.append(raw_range)
        else:
            smoothed.append(raw_range)

    return smoothed


def scan_to_ordered_local_points_with_indices(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    max_radius,
    min_radius=0.10,
    angle_offset_rad=0.0,
    downsample_step=1,
):
    """
    Convert LaserScan fields into ordered local point dictionaries.

    The output preserves scan order and keeps scan metadata next to the local
    Cartesian point so downstream geometry can reason about continuity.
    """
    step = _coerce_step(downsample_step, 1)

    angle_min = float(angle_min)
    angle_increment = float(angle_increment)
    range_min = float(range_min)
    range_max = float(range_max)
    max_radius = float(max_radius)
    min_radius = float(min_radius)
    angle_offset_rad = float(angle_offset_rad)

    ordered_points = []

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

        ordered_points.append(
            {
                "scan_index": i,
                "range": r,
                "scan_angle": scan_angle,
                "local_angle": local_angle,
                "x": x_local,
                "y": y_local,
            }
        )

    return ordered_points


def compute_local_geometry_features(
    ordered_points,
    neighbor_step=1,
    curvature_window=1,
):
    """
    Estimate local neighbor distances, tangent angle, curvature, and changes.
    """
    tangent_step = _coerce_step(neighbor_step, 1)
    curvature_step = _coerce_step(curvature_window, 1)
    count = len(ordered_points)
    features = []

    for index, point in enumerate(ordered_points):
        if index > 0:
            neighbor_distance_prev = _distance_points(
                ordered_points[index - 1],
                point,
            )
        else:
            neighbor_distance_prev = None

        if index + 1 < count:
            neighbor_distance_next = _distance_points(
                point,
                ordered_points[index + 1],
            )
        else:
            neighbor_distance_next = None

        tangent_prev_index = index - tangent_step
        tangent_next_index = index + tangent_step
        if tangent_prev_index >= 0 and tangent_next_index < count:
            tangent_prev = ordered_points[tangent_prev_index]
            tangent_next = ordered_points[tangent_next_index]
        elif index + 1 < count:
            tangent_prev = point
            tangent_next = ordered_points[index + 1]
        elif index > 0:
            tangent_prev = ordered_points[index - 1]
            tangent_next = point
        else:
            tangent_prev = point
            tangent_next = point

        dx = float(tangent_next["x"]) - float(tangent_prev["x"])
        dy = float(tangent_next["y"]) - float(tangent_prev["y"])
        if abs(dx) <= EPSILON and abs(dy) <= EPSILON:
            tangent_angle = 0.0
        else:
            tangent_angle = math.atan2(dy, dx)

        curvature = 0.0
        curvature_valid = False
        curvature_prev_index = index - curvature_step
        curvature_next_index = index + curvature_step
        if curvature_prev_index >= 0 and curvature_next_index < count:
            p_prev = ordered_points[curvature_prev_index]
            p_i = point
            p_next = ordered_points[curvature_next_index]

            prev_x, prev_y = _point_xy(p_prev)
            i_x, i_y = _point_xy(p_i)
            next_x, next_y = _point_xy(p_next)

            a = _distance_xy(prev_x, prev_y, i_x, i_y)
            b = _distance_xy(i_x, i_y, next_x, next_y)
            c = _distance_xy(prev_x, prev_y, next_x, next_y)
            denominator = a * b * c

            if denominator > EPSILON:
                area2 = (
                    (i_x - prev_x) * (next_y - prev_y)
                    - (i_y - prev_y) * (next_x - prev_x)
                )
                curvature = abs(2.0 * area2) / max(denominator, EPSILON)
                curvature_valid = True

        features.append(
            {
                "scan_index": point.get("scan_index"),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "range": float(point["range"]),
                "local_angle": float(point["local_angle"]),
                "neighbor_distance_prev": neighbor_distance_prev,
                "neighbor_distance_next": neighbor_distance_next,
                "tangent_angle": tangent_angle,
                "tangent_angle_delta": 0.0,
                "curvature": curvature,
                "curvature_valid": curvature_valid,
                "curvature_delta": 0.0,
            }
        )

    for index, feature in enumerate(features):
        tangent_deltas = []
        if index > 0:
            tangent_deltas.append(
                abs(
                    normalize_angle(
                        feature["tangent_angle"] - features[index - 1]["tangent_angle"]
                    )
                )
            )
        if index + 1 < count:
            tangent_deltas.append(
                abs(
                    normalize_angle(
                        feature["tangent_angle"] - features[index + 1]["tangent_angle"]
                    )
                )
            )
        if tangent_deltas:
            feature["tangent_angle_delta"] = max(tangent_deltas)

        curvature_deltas = []
        if index > 0:
            curvature_deltas.append(
                abs(feature["curvature"] - features[index - 1]["curvature"])
            )
        if index + 1 < count:
            curvature_deltas.append(
                abs(feature["curvature"] - features[index + 1]["curvature"])
            )
        if curvature_deltas:
            feature["curvature_delta"] = max(curvature_deltas)

    return features


def detect_edge_points(
    features,
    max_neighbor_distance=0.15,
    tangent_jump_threshold_rad=0.55,
    curvature_threshold=2.5,
    curvature_delta_threshold=2.0,
):
    edge_flags = []
    max_neighbor_distance = float(max_neighbor_distance)
    tangent_jump_threshold_rad = float(tangent_jump_threshold_rad)
    curvature_threshold = float(curvature_threshold)
    curvature_delta_threshold = float(curvature_delta_threshold)

    for feature in features:
        is_edge = False

        neighbor_distance_prev = feature.get("neighbor_distance_prev")
        neighbor_distance_next = feature.get("neighbor_distance_next")
        if (
            neighbor_distance_prev is not None
            and neighbor_distance_prev > max_neighbor_distance
        ):
            is_edge = True
        if (
            neighbor_distance_next is not None
            and neighbor_distance_next > max_neighbor_distance
        ):
            is_edge = True

        if feature.get("tangent_angle_delta", 0.0) > tangent_jump_threshold_rad:
            is_edge = True

        if (
            feature.get("curvature", 0.0) > curvature_threshold
            and feature.get("curvature_delta", 0.0) > curvature_delta_threshold
        ):
            is_edge = True

        edge_flags.append(bool(is_edge))

    return edge_flags


def segment_points_by_edges(
    ordered_points,
    features,
    edge_flags,
    max_neighbor_distance=0.15,
    min_segment_size=2,
):
    segments = []
    if not ordered_points:
        return segments

    max_neighbor_distance = float(max_neighbor_distance)
    min_segment_size = _coerce_step(min_segment_size, 1)

    def append_segment(points, segment_features, start_edge, end_edge):
        if not points:
            return
        segments.append(
            {
                "points": list(points),
                "features": list(segment_features),
                "start_edge": bool(start_edge),
                "end_edge": bool(end_edge),
            }
        )

    current_points = [ordered_points[0]]
    current_features = [_copy_feature_with_edge(features[0], edge_flags[0])]
    current_start_edge = bool(edge_flags[0])

    for index in range(1, len(ordered_points)):
        previous_edge = bool(edge_flags[index - 1])
        current_edge = bool(edge_flags[index])
        previous_distance = features[index].get("neighbor_distance_prev")
        if previous_distance is None:
            previous_distance = _distance_points(ordered_points[index - 1], ordered_points[index])

        if previous_distance > max_neighbor_distance:
            append_segment(
                current_points,
                current_features,
                current_start_edge,
                previous_edge,
            )
            current_points = [ordered_points[index]]
            current_features = [_copy_feature_with_edge(features[index], current_edge)]
            current_start_edge = current_edge
            continue

        if previous_edge and len(current_points) >= min_segment_size:
            append_segment(
                current_points,
                current_features,
                current_start_edge,
                True,
            )
            current_points = [ordered_points[index]]
            current_features = [_copy_feature_with_edge(features[index], current_edge)]
            current_start_edge = current_edge
            continue

        current_points.append(ordered_points[index])
        current_features.append(_copy_feature_with_edge(features[index], current_edge))

        if current_edge and len(current_points) >= min_segment_size:
            append_segment(
                current_points,
                current_features,
                current_start_edge,
                True,
            )
            current_points = [ordered_points[index]]
            current_features = [_copy_feature_with_edge(features[index], current_edge)]
            current_start_edge = True

    append_segment(
        current_points,
        current_features,
        current_start_edge,
        bool(edge_flags[-1]),
    )
    return segments


def cluster_points_by_neighbor_distance(
    ordered_points,
    max_neighbor_distance=0.15,
    min_cluster_size=2,
):
    clusters = []
    if not ordered_points:
        return clusters

    max_neighbor_distance = float(max_neighbor_distance)
    min_cluster_size = _coerce_step(min_cluster_size, 1)
    current = [ordered_points[0]]

    for index in range(1, len(ordered_points)):
        previous = ordered_points[index - 1]
        point = ordered_points[index]
        if _distance_points(previous, point) > max_neighbor_distance:
            if len(current) >= min_cluster_size:
                clusters.append(list(current))
            current = [point]
        else:
            current.append(point)

    if len(current) >= min_cluster_size:
        clusters.append(list(current))
    return clusters


def line_candidate_rejection_reason(
    line_fit,
    line_error_threshold,
    min_line_length,
):
    if line_fit is None:
        return "fit_failed"
    if line_fit["length"] < float(min_line_length):
        return "line_too_short"
    if line_fit["mean_error"] > float(line_error_threshold):
        return "mean_error_high"
    if line_fit["max_error"] > float(line_error_threshold) * 2.75 + 0.01:
        return "max_error_high"
    return "accepted"


def make_line_surface_segment_info(points, features, line_fit):
    return {
        "type": "line_surface",
        "line": line_fit,
        "circle": None,
        "mean_curvature": 0.0,
        "std_curvature": 0.0,
        "edge_count": 0,
        "point_count": len(points),
        "points": list(points),
        "features": list(features),
        "start_edge": False,
        "end_edge": False,
    }


def fit_line_segment_from_ordered_points(points):
    if points is None or len(points) < 2:
        return None

    point_count = len(points)
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    cx = sum(xs) / float(point_count)
    cy = sum(ys) / float(point_count)

    cov_xx = 0.0
    cov_xy = 0.0
    cov_yy = 0.0
    for index in range(point_count):
        dx = xs[index] - cx
        dy = ys[index] - cy
        cov_xx += dx * dx
        cov_xy += dx * dy
        cov_yy += dy * dy
    cov_xx /= float(point_count)
    cov_xy /= float(point_count)
    cov_yy /= float(point_count)

    trace = cov_xx + cov_yy
    root = math.sqrt((cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy)
    lambda_major = 0.5 * (trace + root)

    if abs(cov_xy) > EPSILON or abs(lambda_major - cov_xx) > EPSILON:
        direction_x = cov_xy
        direction_y = lambda_major - cov_xx
    elif cov_xx >= cov_yy:
        direction_x = 1.0
        direction_y = 0.0
    else:
        direction_x = 0.0
        direction_y = 1.0

    norm = math.hypot(direction_x, direction_y)
    if norm <= EPSILON:
        return None

    direction_x /= norm
    direction_y /= norm

    projections = []
    errors = []
    for index in range(point_count):
        rel_x = xs[index] - cx
        rel_y = ys[index] - cy
        projection = rel_x * direction_x + rel_y * direction_y
        projections.append(projection)
        errors.append(abs(rel_x * direction_y - rel_y * direction_x))

    min_projection = min(projections)
    max_projection = max(projections)
    sx = cx + min_projection * direction_x
    sy = cy + min_projection * direction_y
    ex = cx + max_projection * direction_x
    ey = cy + max_projection * direction_y
    length = max_projection - min_projection

    return {
        "kind": "line",
        "center": (cx, cy),
        "direction": (direction_x, direction_y),
        "start": (sx, sy),
        "end": (ex, ey),
        "length": abs(length),
        "mean_error": sum(errors) / float(len(errors)),
        "max_error": max(errors),
        "point_count": point_count,
    }


def fit_circle_from_ordered_points(points):
    if points is None or len(points) < 3:
        return None

    point_count = len(points)
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    mean_x = sum(xs) / float(point_count)
    mean_y = sum(ys) / float(point_count)

    suu = 0.0
    suv = 0.0
    svv = 0.0
    su = 0.0
    sv = 0.0
    sw = 0.0
    suw = 0.0
    svw = 0.0

    for index in range(point_count):
        u = xs[index] - mean_x
        v = ys[index] - mean_y
        w = u * u + v * v
        suu += u * u
        suv += u * v
        svv += v * v
        su += u
        sv += v
        sw += w
        suw += u * w
        svw += v * w

    solution = _solve_3x3(
        [
            [suu, suv, su],
            [suv, svv, sv],
            [su, sv, float(point_count)],
        ],
        [suw, svw, sw],
    )
    if solution is None:
        return None

    a_value, b_value, c_value = solution
    center_x = mean_x + 0.5 * a_value
    center_y = mean_y + 0.5 * b_value
    radius_sq = c_value + 0.25 * (a_value * a_value + b_value * b_value)
    if radius_sq <= EPSILON:
        return None

    radius = math.sqrt(radius_sq)
    errors = []
    angles = []
    for index in range(point_count):
        dx = xs[index] - center_x
        dy = ys[index] - center_y
        distance = math.hypot(dx, dy)
        errors.append(abs(distance - radius))
        angles.append(math.atan2(dy, dx))

    return {
        "kind": "circle",
        "center": (center_x, center_y),
        "radius": radius,
        "mean_error": sum(errors) / float(len(errors)),
        "max_error": max(errors),
        "arc_angle": _angle_coverage(angles),
        "point_count": point_count,
    }


def classify_segment_geometry(
    segment,
    line_error_threshold=0.035,
    line_curvature_threshold=0.35,
    min_line_length=0.15,
    circle_error_threshold=0.035,
    circle_curvature_std_threshold=0.60,
    min_circle_radius=0.04,
    max_circle_radius=0.60,
    min_circle_arc_angle=0.35,
):
    points = segment.get("points", [])
    features = segment.get("features", [])
    point_count = len(points)

    curvatures = []
    edge_count = 0
    for feature in features:
        if feature.get("curvature_valid", True):
            curvatures.append(float(feature.get("curvature", 0.0)))
        if feature.get("is_edge"):
            edge_count += 1

    if not curvatures:
        curvatures = [0.0]

    mean_curvature, std_curvature = _mean_std(curvatures)
    line_fit = fit_line_segment_from_ordered_points(points)
    circle_fit = fit_circle_from_ordered_points(points)

    result = {
        "type": "irregular",
        "line": line_fit,
        "circle": circle_fit,
        "mean_curvature": mean_curvature,
        "std_curvature": std_curvature,
        "edge_count": edge_count,
        "point_count": point_count,
        "points": list(points),
        "features": list(features),
        "start_edge": bool(segment.get("start_edge", False)),
        "end_edge": bool(segment.get("end_edge", False)),
    }

    if point_count <= 0:
        return result

    if point_count < 3:
        if edge_count > 0:
            result["type"] = "edge_corner"
        else:
            result["type"] = "irregular"
        return result

    edge_density = float(edge_count) / float(max(1, point_count))

    line_ok = False
    if line_fit is not None:
        line_ok = (
            line_fit["mean_error"] <= float(line_error_threshold)
            and line_fit["length"] >= float(min_line_length)
            and (
                abs(mean_curvature) <= float(line_curvature_threshold)
                or std_curvature <= float(line_curvature_threshold)
                or line_fit["mean_error"] <= float(line_error_threshold) * 0.50
            )
            and edge_density < 0.50
        )

    circle_ok = False
    if circle_fit is not None:
        circle_ok = (
            circle_fit["mean_error"] <= float(circle_error_threshold)
            and circle_fit["radius"] >= float(min_circle_radius)
            and circle_fit["radius"] <= float(max_circle_radius)
            and circle_fit["arc_angle"] >= float(min_circle_arc_angle)
            and std_curvature <= float(circle_curvature_std_threshold)
            and edge_density < 0.50
        )

    if line_ok and circle_ok:
        circle_is_clearer = (
            circle_fit["arc_angle"] >= max(float(min_circle_arc_angle), 0.50)
            and circle_fit["mean_error"] <= line_fit["mean_error"] * 0.55 + 0.003
            and circle_fit["max_error"] <= line_fit["max_error"] * 0.75 + 0.005
        )
        if circle_is_clearer:
            result["type"] = "circle_obstacle"
        else:
            result["type"] = "line_surface"
        return result

    if line_ok:
        result["type"] = "line_surface"
        return result

    if circle_ok:
        result["type"] = "circle_obstacle"
        return result

    if edge_count > 0 and (edge_density >= 0.25 or point_count <= 5):
        result["type"] = "edge_corner"
    else:
        result["type"] = "irregular"
    return result


def geometry_segment_to_circular_obstacles(
    segment_info,
    obstacle_radius=0.08,
    line_sample_spacing=0.22,
    edge_corner_radius_scale=1.5,
    irregular_max_points=3,
    obstacle_inflation=0.0,
    line_max_obstacles=3,
    circle_max_obstacles=2,
):
    obstacle_radius = float(obstacle_radius)
    obstacle_inflation = max(0.0, float(obstacle_inflation))
    inflated_obstacle_radius = obstacle_radius + obstacle_inflation
    segment_type = segment_info.get("type", "irregular")
    obstacles = []

    if segment_type == "line_surface" and segment_info.get("line") is not None:
        line = segment_info["line"]
        start_x, start_y = line["start"]
        end_x, end_y = line["end"]
        length = float(line["length"])
        spacing = max(float(line_sample_spacing), EPSILON)

        if length <= EPSILON:
            center_x, center_y = line["center"]
            return [(center_x, center_y, inflated_obstacle_radius)]

        try:
            line_max_obstacles = int(line_max_obstacles)
        except (TypeError, ValueError):
            line_max_obstacles = 3
        line_max_obstacles = max(2, line_max_obstacles)

        sample_count = int(math.ceil(length / spacing)) + 1
        if sample_count < 2:
            sample_count = 2
        if sample_count > line_max_obstacles:
            sample_count = line_max_obstacles

        for sample_index in range(sample_count):
            ratio = float(sample_index) / float(sample_count - 1)
            x_local = start_x + ratio * (end_x - start_x)
            y_local = start_y + ratio * (end_y - start_y)
            obstacles.append((x_local, y_local, inflated_obstacle_radius))
        return obstacles

    if segment_type == "circle_obstacle" and segment_info.get("circle") is not None:
        circle = segment_info["circle"]
        center_x, center_y = circle["center"]
        radius = max(obstacle_radius, float(circle["radius"])) + obstacle_inflation
        return [(center_x, center_y, radius)]

    if segment_type == "edge_corner":
        edge_radius = inflated_obstacle_radius * float(edge_corner_radius_scale)
        edge_points = []
        points = segment_info.get("points", [])
        features = segment_info.get("features", [])
        for index, feature in enumerate(features):
            if feature.get("is_edge") and index < len(points):
                edge_points.append(points[index])

        if not edge_points:
            edge_points = _nearest_points(points, min(3, len(points)))
        else:
            edge_points = _nearest_points(edge_points, min(3, len(edge_points)))

        for point in edge_points:
            obstacles.append((float(point["x"]), float(point["y"]), edge_radius))
        return obstacles

    points = segment_info.get("points", [])
    for point in _nearest_points(points, irregular_max_points):
        obstacles.append((float(point["x"]), float(point["y"]), inflated_obstacle_radius))
    return obstacles


def _local_obstacles_to_experiment_obstacles(local_obstacles, current_state_exp):
    local_centers = []
    radii = []
    for obstacle in local_obstacles:
        local_centers.append((float(obstacle[0]), float(obstacle[1])))
        radii.append(float(obstacle[2]))

    experiment_centers = local_points_to_experiment_points(
        local_points=local_centers,
        current_state_exp=current_state_exp,
    )

    experiment_obstacles = []
    for index, center in enumerate(experiment_centers):
        experiment_obstacles.append((center[0], center[1], radii[index]))
    return experiment_obstacles


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


def scan_to_experiment_obstacles_geometric(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    current_state_exp,
    max_radius,
    min_radius=0.10,
    angle_offset_rad=0.0,
    downsample_step=1,
    smooth_window_size=3,
    obstacle_radius=0.08,
    max_neighbor_distance=0.15,
    tangent_jump_threshold_rad=0.55,
    curvature_threshold=2.5,
    curvature_delta_threshold=2.0,
    line_error_threshold=0.035,
    line_curvature_threshold=0.35,
    min_line_length=0.15,
    circle_error_threshold=0.035,
    circle_curvature_std_threshold=0.60,
    min_circle_radius=0.04,
    max_circle_radius=0.60,
    min_circle_arc_angle=0.35,
    line_sample_spacing=0.22,
    line_max_obstacles=3,
    edge_corner_radius_scale=1.5,
    irregular_max_points=3,
    max_obstacle_count=80,
    min_segment_points=4,
    obstacle_inflation=0.0,
    return_debug=False,
):
    try:
        line_max_obstacles = int(line_max_obstacles)
    except (TypeError, ValueError):
        line_max_obstacles = 3
    line_max_obstacles = max(2, line_max_obstacles)

    smoothed_ranges = smooth_ranges(
        ranges=ranges,
        window_size=smooth_window_size,
        preserve_invalid=True,
    )
    ordered_points = scan_to_ordered_local_points_with_indices(
        ranges=smoothed_ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        max_radius=max_radius,
        min_radius=min_radius,
        angle_offset_rad=angle_offset_rad,
        downsample_step=downsample_step,
    )
    features = compute_local_geometry_features(
        ordered_points=ordered_points,
        neighbor_step=1,
        curvature_window=2,
    )
    features_by_scan_index = {}
    for index, feature in enumerate(features):
        features_by_scan_index[ordered_points[index].get("scan_index")] = feature

    line_clusters = cluster_points_by_neighbor_distance(
        ordered_points=ordered_points,
        max_neighbor_distance=max_neighbor_distance,
        min_cluster_size=max(2, int(min_segment_points)),
    )
    preclassified_line_infos = []
    consumed_scan_indices = {}
    line_fit_candidates = 0
    line_fit_accepted = 0
    line_rejection_counts = {}
    relaxed_line_error_threshold = float(line_error_threshold) * 1.35
    for cluster in line_clusters:
        if len(cluster) < int(min_segment_points):
            continue
        line_fit_candidates += 1
        line_fit = fit_line_segment_from_ordered_points(cluster)
        rejection_reason = line_candidate_rejection_reason(
            line_fit,
            relaxed_line_error_threshold,
            min_line_length,
        )
        if rejection_reason == "accepted":
            cluster_features = []
            for point in cluster:
                feature = features_by_scan_index.get(point.get("scan_index"))
                if feature is not None:
                    cluster_features.append(_copy_feature_with_edge(feature, False))
            preclassified_line_infos.append(
                make_line_surface_segment_info(cluster, cluster_features, line_fit)
            )
            line_fit_accepted += 1
            for point in cluster:
                consumed_scan_indices[point.get("scan_index")] = True
        else:
            line_rejection_counts[rejection_reason] = (
                line_rejection_counts.get(rejection_reason, 0) + 1
            )

    remaining_points = []
    remaining_features = []
    for index, point in enumerate(ordered_points):
        if consumed_scan_indices.get(point.get("scan_index"), False):
            continue
        remaining_points.append(point)
        remaining_features.append(features[index])

    edge_flags = detect_edge_points(
        features=remaining_features,
        max_neighbor_distance=max_neighbor_distance,
        tangent_jump_threshold_rad=tangent_jump_threshold_rad,
        curvature_threshold=curvature_threshold,
        curvature_delta_threshold=curvature_delta_threshold,
    )
    segments = segment_points_by_edges(
        ordered_points=remaining_points,
        features=remaining_features,
        edge_flags=edge_flags,
        max_neighbor_distance=max_neighbor_distance,
        min_segment_size=max(2, int(min_segment_points)),
    )

    segment_infos = list(preclassified_line_infos)
    local_obstacles = []
    compressed_line_obstacles = 0
    counts_by_type = {
        "line_surface": 0,
        "circle_obstacle": 0,
        "edge_corner": 0,
        "irregular": 0,
    }
    counts_by_type["line_surface"] = len(preclassified_line_infos)

    for segment_info in preclassified_line_infos:
        segment_obstacles = geometry_segment_to_circular_obstacles(
            segment_info=segment_info,
            obstacle_radius=obstacle_radius,
            line_sample_spacing=line_sample_spacing,
            line_max_obstacles=line_max_obstacles,
            edge_corner_radius_scale=edge_corner_radius_scale,
            irregular_max_points=irregular_max_points,
            obstacle_inflation=obstacle_inflation,
        )
        compressed_line_obstacles += len(segment_obstacles)
        local_obstacles.extend(segment_obstacles)

    for segment in segments:
        if len(segment.get("points", [])) < int(min_segment_points):
            segment_info = {
                "type": "edge_corner" if segment.get("start_edge") or segment.get("end_edge") else "irregular",
                "line": None,
                "circle": None,
                "mean_curvature": 0.0,
                "std_curvature": 0.0,
                "edge_count": 0,
                "point_count": len(segment.get("points", [])),
                "points": list(segment.get("points", [])),
                "features": list(segment.get("features", [])),
                "start_edge": bool(segment.get("start_edge", False)),
                "end_edge": bool(segment.get("end_edge", False)),
            }
        else:
            segment_info = classify_segment_geometry(
                segment=segment,
                line_error_threshold=line_error_threshold,
                line_curvature_threshold=line_curvature_threshold,
                min_line_length=min_line_length,
                circle_error_threshold=circle_error_threshold,
                circle_curvature_std_threshold=circle_curvature_std_threshold,
                min_circle_radius=min_circle_radius,
                max_circle_radius=max_circle_radius,
                min_circle_arc_angle=min_circle_arc_angle,
            )
        segment_infos.append(segment_info)
        segment_type = segment_info.get("type", "irregular")
        if segment_type not in counts_by_type:
            segment_type = "irregular"
        counts_by_type[segment_type] += 1

        segment_obstacles = geometry_segment_to_circular_obstacles(
            segment_info=segment_info,
            obstacle_radius=obstacle_radius,
            line_sample_spacing=line_sample_spacing,
            line_max_obstacles=line_max_obstacles,
            edge_corner_radius_scale=edge_corner_radius_scale,
            irregular_max_points=irregular_max_points,
            obstacle_inflation=obstacle_inflation,
        )
        if segment_type == "line_surface":
            compressed_line_obstacles += len(segment_obstacles)
        local_obstacles.extend(segment_obstacles)

    experiment_obstacles = _local_obstacles_to_experiment_obstacles(
        local_obstacles=local_obstacles,
        current_state_exp=current_state_exp,
    )

    experiment_obstacles_before_limit = list(experiment_obstacles)
    obstacle_count_before_limit = len(experiment_obstacles)
    try:
        max_obstacle_count = int(max_obstacle_count)
    except (TypeError, ValueError):
        max_obstacle_count = 0
    experiment_obstacles = limit_obstacles_evenly(
        experiment_obstacles,
        max_obstacle_count,
    )
    rejection_parts = []
    for key in sorted(line_rejection_counts.keys()):
        rejection_parts.append("{}:{}".format(key, line_rejection_counts[key]))
    if not rejection_parts:
        line_fit_rejected_reason = "none"
    else:
        line_fit_rejected_reason = ",".join(rejection_parts)

    debug = {
        "mode": "geometric",
        "local_point_count": len(ordered_points),
        "feature_count": len(features),
        "edge_count": sum(1 for flag in edge_flags if flag),
        "segment_count": len(segments) + len(preclassified_line_infos),
        "line_surface_count": counts_by_type["line_surface"],
        "circle_obstacle_count": counts_by_type["circle_obstacle"],
        "edge_corner_count": counts_by_type["edge_corner"],
        "irregular_count": counts_by_type["irregular"],
        "obstacle_count_before_limit": obstacle_count_before_limit,
        "obstacle_count_after_limit": len(experiment_obstacles),
        "geometric_obstacle_count_raw": obstacle_count_before_limit,
        "geometric_obstacle_count_used": len(experiment_obstacles),
        "compressed_line_obstacles": compressed_line_obstacles,
        "line_fit_candidates": line_fit_candidates,
        "line_fit_accepted": line_fit_accepted,
        "line_fit_rejected_reason": line_fit_rejected_reason,
        "line_surface_streak": 1 if counts_by_type["line_surface"] > 0 else 0,
        "representative_circle_count": compressed_line_obstacles,
        "obstacle_budget_mode": "geometric_even_limit_{}".format(max_obstacle_count),
        "line_compression_mode": (
            "representative_circles_2_3"
            if counts_by_type["line_surface"] > 0
            else "none"
        ),
        "max_obstacle_count": max_obstacle_count,
        "min_segment_points": int(min_segment_points),
        "obstacle_inflation": float(obstacle_inflation),
        "line_max_obstacles": int(line_max_obstacles),
        "experiment_obstacles_before_limit": list(experiment_obstacles_before_limit),
        "experiment_obstacles_after_limit": list(experiment_obstacles),
    }

    if return_debug:
        return experiment_obstacles, debug
    return experiment_obstacles


def _make_ordered_points_from_xy(xy_points):
    ordered_points = []
    for index, xy in enumerate(xy_points):
        x_value = float(xy[0])
        y_value = float(xy[1])
        point_range = math.hypot(x_value, y_value)
        local_angle = math.atan2(y_value, x_value)
        ordered_points.append(
            {
                "scan_index": index,
                "range": point_range,
                "scan_angle": local_angle,
                "local_angle": local_angle,
                "x": x_value,
                "y": y_value,
            }
        )
    return ordered_points


def _segment_from_points(points, max_neighbor_distance=0.15):
    features = compute_local_geometry_features(points, curvature_window=2)
    edge_flags = detect_edge_points(
        features,
        max_neighbor_distance=max_neighbor_distance,
    )
    segment_features = []
    for index, feature in enumerate(features):
        segment_features.append(_copy_feature_with_edge(feature, edge_flags[index]))
    return (
        {
            "points": points,
            "features": segment_features,
            "start_edge": bool(edge_flags[0]) if edge_flags else False,
            "end_edge": bool(edge_flags[-1]) if edge_flags else False,
        },
        features,
        edge_flags,
    )


def _make_wall_scan_360():
    ranges = [float("inf")] * 360
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / 360.0
    for index in range(31):
        y_value = -0.45 + 0.90 * float(index) / 30.0
        x_value = 1.0
        angle = math.atan2(y_value, x_value)
        scan_index = int(round((angle - angle_min) / angle_increment))
        if scan_index < 0 or scan_index >= len(ranges):
            continue
        point_range = math.hypot(x_value, y_value)
        if (
            not is_finite_number(ranges[scan_index])
            or point_range < ranges[scan_index]
        ):
            ranges[scan_index] = point_range
    return ranges, angle_min, angle_increment


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

    smoothed = smooth_ranges(
        ranges=[1.0, 1.2, float("inf"), 1.4, float("nan")],
        window_size=3,
    )
    print("smoothed ranges:", smoothed)
    assert is_finite_number(smoothed[0])
    assert not is_finite_number(smoothed[2])
    assert not is_finite_number(smoothed[4])

    ordered = scan_to_ordered_local_points_with_indices(
        ranges=[1.0, 1.0, 1.0, 1.0, 1.0],
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.05,
        range_max=6.0,
        max_radius=2.0,
        downsample_step=2,
    )
    points = [(point["x"], point["y"]) for point in ordered]
    print("downsampled ordered points:", points)
    assert len(points) == 3
    assert ordered[0]["scan_index"] == 0
    assert ordered[1]["scan_index"] == 2

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
    print("raw experiment obstacles:", obstacles)
    assert len(obstacles) == 1
    assert abs(obstacles[0][0] - 2.0) < 1e-6
    assert abs(obstacles[0][1] - 2.0) < 1e-6
    assert abs(obstacles[0][2] - 0.08) < 1e-9

    many_obstacles = []
    for index in range(10):
        many_obstacles.append((float(index), 0.0, 0.08))
    limited_obstacles = limit_obstacles_evenly(many_obstacles, 4)
    limited_indices = [int(obstacle[0]) for obstacle in limited_obstacles]
    unlimited_obstacles = limit_obstacles_evenly(many_obstacles, 0)
    print("evenly limited obstacles:", limited_indices)
    assert len(limited_obstacles) == 4
    assert limited_indices[0] == 0
    assert limited_indices[-1] == 9
    assert limited_indices != [0, 1, 2, 3]
    assert limited_indices[1] > 1
    assert limited_indices[2] < 8
    assert len(unlimited_obstacles) == len(many_obstacles)
    assert unlimited_obstacles == many_obstacles
    assert unlimited_obstacles is not many_obstacles

    line_xy = []
    for index in range(25):
        y_value = -0.45 + 0.90 * float(index) / 24.0
        x_value = 1.0 + ((index % 3) - 1) * 0.002
        line_xy.append((x_value, y_value))
    line_points = _make_ordered_points_from_xy(line_xy)
    line_segment, line_features, line_edge_flags = _segment_from_points(line_points)
    line_info = classify_segment_geometry(line_segment)
    line_obstacles = geometry_segment_to_circular_obstacles(line_info)
    print(
        "line surface type={}, points={}, obstacles={}".format(
            line_info["type"],
            len(line_points),
            len(line_obstacles),
        )
    )
    assert len(line_features) == len(line_points)
    assert len(line_edge_flags) == len(line_points)
    assert line_info["type"] == "line_surface"
    assert len(line_obstacles) < len(line_points)
    assert _distance_xy(
        line_obstacles[0][0],
        line_obstacles[0][1],
        line_obstacles[-1][0],
        line_obstacles[-1][1],
    ) > 0.50

    corner_xy = []
    for index in range(13):
        corner_xy.append((1.0, -0.45 + 0.45 * float(index) / 12.0))
    for index in range(1, 14):
        corner_xy.append((1.0 + 0.45 * float(index) / 13.0, 0.0))
    corner_points = _make_ordered_points_from_xy(corner_xy)
    corner_features = compute_local_geometry_features(corner_points)
    corner_edge_flags = detect_edge_points(
        corner_features,
        max_neighbor_distance=0.16,
        tangent_jump_threshold_rad=0.45,
        curvature_threshold=1.0,
        curvature_delta_threshold=0.5,
    )
    corner_segments = segment_points_by_edges(
        corner_points,
        corner_features,
        corner_edge_flags,
        max_neighbor_distance=0.16,
    )
    corner_infos = [classify_segment_geometry(segment) for segment in corner_segments]
    corner_types = [info["type"] for info in corner_infos]
    print(
        "corner segments={}, edges={}, types={}".format(
            len(corner_segments),
            sum(1 for flag in corner_edge_flags if flag),
            corner_types,
        )
    )
    assert len(corner_segments) >= 2 or "edge_corner" in corner_types

    arc_xy = []
    arc_center = (0.85, 0.05)
    arc_radius = 0.18
    for index in range(29):
        theta = -0.95 + 1.90 * float(index) / 28.0
        arc_xy.append(
            (
                arc_center[0] + arc_radius * math.cos(theta),
                arc_center[1] + arc_radius * math.sin(theta),
            )
        )
    arc_points = _make_ordered_points_from_xy(arc_xy)
    arc_segment, arc_features, arc_edge_flags = _segment_from_points(
        arc_points,
        max_neighbor_distance=0.16,
    )
    arc_info = classify_segment_geometry(arc_segment)
    print(
        "circle arc type={}, radius={:.3f}, mean_error={:.5f}, arc_angle={:.3f}".format(
            arc_info["type"],
            arc_info["circle"]["radius"] if arc_info["circle"] else -1.0,
            arc_info["circle"]["mean_error"] if arc_info["circle"] else -1.0,
            arc_info["circle"]["arc_angle"] if arc_info["circle"] else -1.0,
        )
    )
    assert len(arc_features) == len(arc_points)
    assert len(arc_edge_flags) == len(arc_points)
    assert arc_info["circle"] is not None
    assert abs(arc_info["circle"]["radius"] - arc_radius) < 0.02
    assert arc_info["type"] in ("circle_obstacle", "line_surface")
    if arc_info["type"] != "circle_obstacle":
        print("circle arc classified conservatively as line_surface")

    irregular_xy = [
        (0.55, 0.27),
        (0.72, -0.19),
        (0.61, 0.05),
        (0.94, 0.24),
        (0.83, -0.31),
        (1.07, 0.04),
        (0.66, -0.34),
        (1.12, -0.22),
        (0.91, 0.12),
        (0.78, 0.31),
    ]
    irregular_points = _make_ordered_points_from_xy(irregular_xy)
    irregular_segment, _, _ = _segment_from_points(
        irregular_points,
        max_neighbor_distance=0.50,
    )
    irregular_info = classify_segment_geometry(irregular_segment)
    irregular_obstacles = geometry_segment_to_circular_obstacles(
        irregular_info,
        irregular_max_points=4,
    )
    print(
        "irregular type={}, obstacles={}".format(
            irregular_info["type"],
            len(irregular_obstacles),
        )
    )
    assert irregular_info["type"] in ("irregular", "edge_corner")
    assert len(irregular_obstacles) <= 4 or irregular_info["type"] == "edge_corner"

    jump_xy = []
    for index in range(6):
        jump_xy.append((0.80, -0.20 + 0.02 * float(index)))
    for index in range(6):
        jump_xy.append((0.80, 0.35 + 0.02 * float(index)))
    jump_points = _make_ordered_points_from_xy(jump_xy)
    jump_features = compute_local_geometry_features(jump_points)
    jump_edge_flags = detect_edge_points(
        jump_features,
        max_neighbor_distance=0.15,
    )
    jump_segments = segment_points_by_edges(
        jump_points,
        jump_features,
        jump_edge_flags,
        max_neighbor_distance=0.15,
    )
    jump_edge_count = sum(1 for flag in jump_edge_flags if flag)
    print(
        "jump edge_count={}, segment_count={}".format(
            jump_edge_count,
            len(jump_segments),
        )
    )
    assert jump_edge_count > 0 or len(jump_segments) >= 2

    wall_ranges, wall_angle_min, wall_angle_increment = _make_wall_scan_360()
    geometric_obstacles, geometric_debug = scan_to_experiment_obstacles_geometric(
        ranges=wall_ranges,
        angle_min=wall_angle_min,
        angle_increment=wall_angle_increment,
        range_min=0.05,
        range_max=6.0,
        current_state_exp=(0.0, 0.0, 0.0),
        max_radius=2.0,
        downsample_step=1,
        max_obstacle_count=5,
        return_debug=True,
    )
    print("geometric entry debug:", geometric_debug)
    assert geometric_debug["mode"] == "geometric"
    assert geometric_debug["obstacle_count_after_limit"] <= 5
    assert len(geometric_obstacles) == geometric_debug["obstacle_count_after_limit"]

    performance_ranges = []
    for index in range(360):
        angle = -math.pi + 2.0 * math.pi * float(index) / 360.0
        performance_ranges.append(1.0 + 0.05 * math.sin(3.0 * angle))

    iterations = 20
    raw_start = time.time()
    raw_obstacles = []
    for _ in range(iterations):
        raw_obstacles = scan_to_experiment_obstacles(
            ranges=performance_ranges,
            angle_min=-math.pi,
            angle_increment=2.0 * math.pi / 360.0,
            range_min=0.05,
            range_max=6.0,
            current_state_exp=(0.0, 0.0, 0.0),
            max_radius=2.0,
            downsample_step=1,
            obstacle_radius=0.08,
        )
    raw_elapsed_ms = (time.time() - raw_start) * 1000.0 / float(iterations)

    geometric_start = time.time()
    geometric_perf_obstacles = []
    geometric_perf_debug = {}
    for _ in range(iterations):
        geometric_perf_obstacles, geometric_perf_debug = scan_to_experiment_obstacles_geometric(
            ranges=performance_ranges,
            angle_min=-math.pi,
            angle_increment=2.0 * math.pi / 360.0,
            range_min=0.05,
            range_max=6.0,
            current_state_exp=(0.0, 0.0, 0.0),
            max_radius=2.0,
            downsample_step=1,
            max_obstacle_count=80,
            return_debug=True,
        )
    geometric_elapsed_ms = (
        time.time() - geometric_start
    ) * 1000.0 / float(iterations)

    print(
        "performance raw_360_ms={:.3f}, geometric_360_ms={:.3f}, "
        "raw_obstacles={}, geometric_obstacles={}".format(
            raw_elapsed_ms,
            geometric_elapsed_ms,
            len(raw_obstacles),
            len(geometric_perf_obstacles),
        )
    )
    print("performance geometric debug:", geometric_perf_debug)
    if geometric_elapsed_ms > 50.0:
        print("WARNING: geometric mode exceeded 50 ms in this rough local test.")

    print("")
    print("All local_obstacle_layer basic tests passed.")


if __name__ == "__main__":
    _run_basic_tests()
