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


def analyze_scan_front_sector(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    front_stop_distance,
    front_slow_distance,
    front_angle_deg,
    front_angle_offset_deg=0.0,
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
    front_angle_rad = math.radians(float(front_angle_deg))
    front_angle_offset_rad = math.radians(float(front_angle_offset_deg))

    front_ranges = []

    for i, raw_range in enumerate(ranges):
        if not is_finite_range(raw_range):
            continue

        r = float(raw_range)

        if r < range_min or r > range_max:
            continue

        angle = float(angle_min) + float(i) * float(angle_increment)

        if angle_in_front_cone(
            angle=angle,
            front_angle_rad=front_angle_rad,
            front_angle_offset_rad=front_angle_offset_rad,
        ):
            front_ranges.append(r)

    if not front_ranges:
        return {
            "emergency_stop": True,
            "should_slow_down": False,
            "slow_scale": 0.0,
            "min_front_range": None,
            "valid_front_count": 0,
            "reason": "no_valid_front_scan",
        }

    min_front_range = min(front_ranges)

    if min_front_range <= front_stop_distance:
        return {
            "emergency_stop": True,
            "should_slow_down": False,
            "slow_scale": 0.0,
            "min_front_range": min_front_range,
            "valid_front_count": len(front_ranges),
            "reason": "front_obstacle_stop",
        }

    if min_front_range <= front_slow_distance:
        # 距离越接近 stop_distance，速度越小。
        denom = max(front_slow_distance - front_stop_distance, 1e-6)
        ratio = (min_front_range - front_stop_distance) / denom

        # 最低给 0.25，避免一点点障碍就完全爬不动。
        slow_scale = max(0.25, min(1.0, ratio))

        return {
            "emergency_stop": False,
            "should_slow_down": True,
            "slow_scale": slow_scale,
            "min_front_range": min_front_range,
            "valid_front_count": len(front_ranges),
            "reason": "front_obstacle_slow",
        }

    return {
        "emergency_stop": False,
        "should_slow_down": False,
        "slow_scale": 1.0,
        "min_front_range": min_front_range,
        "valid_front_count": len(front_ranges),
        "reason": "front_clear",
    }


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
