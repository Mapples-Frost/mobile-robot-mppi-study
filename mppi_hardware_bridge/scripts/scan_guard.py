# -*- coding: utf-8 -*-
from __future__ import print_function

import math


def is_finite_range(value):
    """
    判断一个雷达距离值是不是有效数字。

    LaserScan 里可能出现：
        inf
        nan
        0.0
        过大值

    第一版我们只接受有限的正常数字。
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return not (math.isnan(value) or math.isinf(value))


def normalize_angle(angle):
    """
    把角度归一化到 [-pi, pi]。
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_in_front_cone(angle, front_angle_rad, front_angle_offset_rad=0.0):
    """
    判断某个雷达点是否落在“小车正前方扇形区域”内。

    angle:
        当前雷达点相对雷达坐标系的角度，单位 rad。

    front_angle_rad:
        正前方扇形半角，单位 rad。
        例如 35 deg 表示看前方左右各 35 度。

    front_angle_offset_rad:
        雷达正前方相对 angle=0 的偏置。
        第一版默认 0。
        如果现场发现雷达 angle=0 不是车头正前方，可以调这个值。
    """
    relative_angle = normalize_angle(angle - front_angle_offset_rad)
    return abs(relative_angle) <= front_angle_rad


def scan_ranges_to_base_points(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    front_angle_offset_deg=0.0,
):
    """
    Convert finite LaserScan returns to base-frame points using an explicit
    scan angle offset. A point at scan angle == offset is treated as +X/front.
    """
    points = []
    front_angle_offset_rad = math.radians(float(front_angle_offset_deg))
    range_min = float(range_min)
    range_max = float(range_max)

    for i, raw_range in enumerate(ranges):
        if not is_finite_range(raw_range):
            continue

        r = float(raw_range)
        if r < range_min or r > range_max:
            continue

        scan_angle = float(angle_min) + float(i) * float(angle_increment)
        base_angle = normalize_angle(scan_angle - front_angle_offset_rad)
        points.append(
            {
                "index": i,
                "range": r,
                "scan_angle": scan_angle,
                "base_angle": base_angle,
                "x": r * math.cos(base_angle),
                "y": r * math.sin(base_angle),
            }
        )

    return points


def _min_range(points):
    if not points:
        return None
    return min([float(point["range"]) for point in points])


def _side_obstacle_side(left_min_range, right_min_range):
    if left_min_range is None and right_min_range is None:
        return "none"
    if left_min_range is None:
        return "right"
    if right_min_range is None:
        return "left"
    if float(left_min_range) <= float(right_min_range):
        return "left"
    return "right"


def _front_slow_scale(
    min_front_range,
    front_soft_block_distance,
    front_slow_distance,
    min_scale=0.45,
):
    denom = max(float(front_slow_distance) - float(front_soft_block_distance), 1e-6)
    ratio = (float(min_front_range) - float(front_soft_block_distance)) / denom
    min_scale = max(0.0, min(1.0, float(min_scale)))
    return max(min_scale, min(1.0, ratio))


def analyze_scan_front_sector(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    front_stop_distance,
    front_slow_distance,
    front_angle_deg,
    hard_stop_distance=None,
    front_soft_block_distance=None,
    front_slow_min_scale=0.45,
    front_angle_offset_deg=0.0,
    side_stop_distance=0.0,
    side_angle_deg=120.0,
    near_body_stop_radius=0.0,
    side_soft_distance=None,
    side_hard_distance=None,
    side_release_distance=None,
    side_front_exclusion_angle_deg=25.0,
):
    """
    分析 LaserScan 正前方区域是否危险。

    输入：
        ranges:
            雷达距离数组。

        angle_min:
            ranges[0] 对应的角度，单位 rad。

        angle_increment:
            相邻两个雷达点之间的角度间隔，单位 rad。

        range_min / range_max:
            雷达有效距离范围，单位 m。

        front_stop_distance:
            小于这个距离就急停，单位 m。

        front_slow_distance:
            小于这个距离但还没到急停距离，就进入减速区，单位 m。

        front_angle_deg:
            正前方检测扇形半角，单位 deg。

        front_angle_offset_deg:
            雷达正前方偏置角，单位 deg。
            默认 0。

    输出：
        result:
            一个 dict，里面包含：
                emergency_stop
                should_slow_down
                slow_scale
                min_front_range
                valid_front_count
    """
    front_stop_distance = float(front_stop_distance)
    front_slow_distance = float(front_slow_distance)
    if hard_stop_distance is None:
        hard_stop_distance = front_stop_distance
    if front_soft_block_distance is None:
        front_soft_block_distance = front_stop_distance
    hard_stop_distance = float(hard_stop_distance)
    front_soft_block_distance = float(front_soft_block_distance)
    if front_soft_block_distance < hard_stop_distance:
        front_soft_block_distance = hard_stop_distance
    if front_slow_distance < front_soft_block_distance:
        front_slow_distance = front_soft_block_distance
    front_slow_min_scale = max(0.0, min(1.0, float(front_slow_min_scale)))

    front_angle_rad = math.radians(float(front_angle_deg))
    side_angle_rad = math.radians(float(side_angle_deg))
    side_stop_distance = float(side_stop_distance)
    near_body_stop_radius = float(near_body_stop_radius)
    if side_hard_distance is None:
        side_hard_distance = side_stop_distance
    if side_soft_distance is None:
        side_soft_distance = side_stop_distance
    if side_release_distance is None:
        side_release_distance = max(float(side_soft_distance), side_stop_distance)

    side_soft_distance = float(side_soft_distance)
    side_hard_distance = float(side_hard_distance)
    side_release_distance = float(side_release_distance)
    side_front_exclusion_angle_rad = math.radians(
        max(0.0, float(side_front_exclusion_angle_deg))
    )
    if side_front_exclusion_angle_rad > side_angle_rad:
        side_front_exclusion_angle_rad = side_angle_rad
    side_detection_enabled = (
        side_soft_distance > 0.0
        or side_hard_distance > 0.0
        or side_stop_distance > 0.0
    )

    raw_points_base = scan_ranges_to_base_points(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        front_angle_offset_deg=front_angle_offset_deg,
    )

    front_points = []
    side_points = []
    left_side_points = []
    right_side_points = []
    near_body_points = []
    for point in raw_points_base:
        base_angle = float(point["base_angle"])
        if abs(base_angle) <= front_angle_rad:
            front_points.append(point)
        if (
            side_detection_enabled
            and abs(base_angle) > side_front_exclusion_angle_rad
            and abs(base_angle) <= side_angle_rad
        ):
            side_points.append(point)
            if float(point.get("y", 0.0)) >= 0.0:
                left_side_points.append(point)
            else:
                right_side_points.append(point)
        if near_body_stop_radius > 0.0 and float(point["range"]) <= near_body_stop_radius:
            near_body_points.append(point)

    min_front_range = _min_range(front_points)
    min_side_range = _min_range(side_points)
    min_left_side_range = _min_range(left_side_points)
    min_right_side_range = _min_range(right_side_points)
    min_near_body_range = _min_range(near_body_points)
    side_obstacle_side = _side_obstacle_side(
        min_left_side_range,
        min_right_side_range,
    )
    side_hard_active = (
        min_side_range is not None
        and side_hard_distance > 0.0
        and float(min_side_range) < side_hard_distance
    )
    side_soft_active = (
        min_side_range is not None
        and side_soft_distance > 0.0
        and float(min_side_range) < side_soft_distance
        and not side_hard_active
    )

    base_result = {
        "front_stop_mode": "clear",
        "front_stop_distance": front_stop_distance,
        "front_soft_block_distance": front_soft_block_distance,
        "hard_stop_distance": hard_stop_distance,
        "front_slow_min_scale": front_slow_min_scale,
        "min_side_range": min_side_range,
        "min_left_side_range": min_left_side_range,
        "min_right_side_range": min_right_side_range,
        "min_near_body_range": min_near_body_range,
        "valid_side_count": len(side_points),
        "valid_left_side_count": len(left_side_points),
        "valid_right_side_count": len(right_side_points),
        "valid_near_body_count": len(near_body_points),
        "side_points": list(side_points),
        "left_side_points": list(left_side_points),
        "right_side_points": list(right_side_points),
        "near_body_points": list(near_body_points),
        "raw_points_base": list(raw_points_base),
        "side_avoid_mode": "side_soft_avoid" if side_soft_active else "none",
        "side_stop_mode": "side_hard_stop" if side_hard_active else "none",
        "side_obstacle_side": side_obstacle_side,
        "side_soft_distance": side_soft_distance,
        "side_hard_distance": side_hard_distance,
        "side_release_distance": side_release_distance,
        "side_front_exclusion_angle_deg": float(side_front_exclusion_angle_deg),
        "side_avoid_applied": False,
        "side_hard_stop_applied": bool(side_hard_active),
    }

    if near_body_points:
        result = {
            "emergency_stop": True,
            "should_slow_down": False,
            "slow_scale": 0.0,
            "min_front_range": min_front_range,
            "valid_front_count": len(front_points),
            "front_points": list(front_points),
            "reason": "near_body_hard_stop",
        }
        result.update(base_result)
        result["front_stop_mode"] = "near_body_hard_stop"
        result["side_stop_mode"] = "near_body_hard_stop"
        result["side_hard_stop_applied"] = True
        return result

    if not front_points:
        result = {
            "emergency_stop": True,
            "should_slow_down": False,
            "slow_scale": 0.0,
            "min_front_range": None,
            "valid_front_count": 0,
            "front_points": [],
            "reason": "no_valid_front_scan",
        }
        result.update(base_result)
        result["front_stop_mode"] = "no_valid_front_scan"
        return result

    if min_front_range <= hard_stop_distance:
        result = {
            "emergency_stop": True,
            "should_slow_down": False,
            "slow_scale": 0.0,
            "min_front_range": min_front_range,
            "valid_front_count": len(front_points),
            "front_points": list(front_points),
            "reason": "hard_stop",
        }
        result.update(base_result)
        result["front_stop_mode"] = "hard_stop"
        return result

    if side_hard_active:
        result = {
            "emergency_stop": True,
            "should_slow_down": False,
            "slow_scale": 0.0,
            "min_front_range": min_front_range,
            "valid_front_count": len(front_points),
            "front_points": list(front_points),
            "reason": "side_hard_stop",
        }
        result.update(base_result)
        result["front_stop_mode"] = "side_hard_stop"
        return result

    if min_front_range <= front_soft_block_distance:
        result = {
            "emergency_stop": False,
            "should_slow_down": True,
            "slow_scale": 1.0,
            "min_front_range": min_front_range,
            "valid_front_count": len(front_points),
            "front_points": list(front_points),
            "reason": "front_soft_block",
        }
        result.update(base_result)
        result["front_stop_mode"] = "front_soft_block"
        return result

    if min_front_range <= front_slow_distance:
        # 距离越接近 stop_distance，速度越小。
        slow_scale = _front_slow_scale(
            min_front_range,
            front_soft_block_distance,
            front_slow_distance,
            front_slow_min_scale,
        )

        # 最低给 0.25，避免一点点障碍就完全爬不动。

        result = {
            "emergency_stop": False,
            "should_slow_down": True,
            "slow_scale": slow_scale,
            "min_front_range": min_front_range,
            "valid_front_count": len(front_points),
            "front_points": list(front_points),
            "reason": "front_obstacle_slow",
        }
        result.update(base_result)
        result["front_stop_mode"] = "front_obstacle_slow"
        return result

    if side_soft_active:
        result = {
            "emergency_stop": False,
            "should_slow_down": False,
            "slow_scale": 1.0,
            "min_front_range": min_front_range,
            "valid_front_count": len(front_points),
            "front_points": list(front_points),
            "reason": "side_obstacle_soft",
        }
        result.update(base_result)
        result["front_stop_mode"] = "side_obstacle_soft"
        return result

    result = {
        "emergency_stop": False,
        "should_slow_down": False,
        "slow_scale": 1.0,
        "min_front_range": min_front_range,
        "valid_front_count": len(front_points),
        "front_points": list(front_points),
        "reason": "front_clear",
    }
    result.update(base_result)
    return result


def _make_fake_scan(
    num_points=181,
    angle_min=-math.pi / 2.0,
    angle_max=math.pi / 2.0,
    default_range=3.0,
):
    """
    构造一个假的前向 180 度雷达，用于本地测试。

    angle 从 -90 deg 到 +90 deg。
    angle = 0 表示正前方。
    """
    if num_points <= 1:
        raise ValueError("num_points must be greater than 1.")

    angle_increment = (angle_max - angle_min) / float(num_points - 1)
    ranges = [default_range for _ in range(num_points)]

    return ranges, angle_min, angle_increment


def _run_basic_tests():
    """
    本地测试，不依赖 ROS。
    """

    range_min = 0.05
    range_max = 6.0
    front_stop_distance = 0.45
    front_slow_distance = 0.80
    front_angle_deg = 35.0

    print("Test 1: front clear")
    ranges, angle_min, angle_increment = _make_fake_scan(default_range=3.0)

    result = analyze_scan_front_sector(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        front_stop_distance=front_stop_distance,
        front_slow_distance=front_slow_distance,
        front_angle_deg=front_angle_deg,
    )
    print("  result =", result)

    print("")
    print("Test 2: obstacle in front, stop")
    ranges, angle_min, angle_increment = _make_fake_scan(default_range=3.0)

    # 中间点大约是 angle=0，也就是正前方。
    middle = len(ranges) // 2
    ranges[middle] = 0.30

    result = analyze_scan_front_sector(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        front_stop_distance=front_stop_distance,
        front_slow_distance=front_slow_distance,
        front_angle_deg=front_angle_deg,
    )
    print("  result =", result)

    print("")
    print("Test 3: obstacle in front, slow down")
    ranges, angle_min, angle_increment = _make_fake_scan(default_range=3.0)
    ranges[middle] = 0.65

    result = analyze_scan_front_sector(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        front_stop_distance=front_stop_distance,
        front_slow_distance=front_slow_distance,
        front_angle_deg=front_angle_deg,
    )
    print("  result =", result)

    print("")
    print("Test 4: obstacle on side, should not stop")
    ranges, angle_min, angle_increment = _make_fake_scan(default_range=3.0)

    # 最左侧约 -90 deg，不在 front_angle_deg=35 deg 的正前方扇形里。
    ranges[0] = 0.20

    result = analyze_scan_front_sector(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        front_stop_distance=front_stop_distance,
        front_slow_distance=front_slow_distance,
        front_angle_deg=front_angle_deg,
    )
    print("  result =", result)

    print("")
    print("Test 5: no valid front scan, fail safe stop")
    ranges, angle_min, angle_increment = _make_fake_scan(default_range=float("inf"))

    result = analyze_scan_front_sector(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        front_stop_distance=front_stop_distance,
        front_slow_distance=front_slow_distance,
        front_angle_deg=front_angle_deg,
    )
    print("  result =", result)


if __name__ == "__main__":
    _run_basic_tests()
