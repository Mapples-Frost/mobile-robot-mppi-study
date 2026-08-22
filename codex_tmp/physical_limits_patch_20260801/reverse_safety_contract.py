"""Pure signed-speed contract shared by the laptop and ROS gateway.

This module deliberately has no ROS, NumPy, or CUDA dependency so the exact
physical command semantics can be regression-tested on the laptop and loaded
by the robot's Python 2 ROS Kinetic runtime.
"""


def clamp(value, low, high):
    return min(max(float(value), float(low)), float(high))


def bound_signed_speed(
    value,
    max_forward_mps,
    max_reverse_mps,
    allow_reverse=True,
):
    low = -float(max_reverse_mps) if allow_reverse else 0.0
    return clamp(value, low, max_forward_mps)


def couple_differential_drive_command(
    linear_mps,
    angular_radps,
    wheel_track_m,
    maximum_wheel_speed_mps,
):
    """Scale a twist so neither differential-drive wheel is over-commanded."""
    linear_mps = float(linear_mps)
    angular_radps = float(angular_radps)
    half_track = 0.5 * float(wheel_track_m)
    wheel_limit = float(maximum_wheel_speed_mps)
    if half_track <= 0.0 or wheel_limit <= 0.0:
        raise ValueError("wheel geometry and speed limit must be positive")
    right = linear_mps + angular_radps * half_track
    left = linear_mps - angular_radps * half_track
    peak = max(abs(right), abs(left))
    if peak <= wheel_limit:
        return linear_mps, angular_radps, False
    scale = wheel_limit / peak
    return linear_mps * scale, angular_radps * scale, True


def protected_reverse_escape(
    requested_linear_mps,
    requested_angular_radps,
    maximum_reverse_mps,
    rear_clearance_m,
    rear_slow_m,
    rear_sensors_fresh,
    dynamic_threat_confirmed,
    fallback_reverse_mps=0.05,
):
    """Recover from a front near-body stop only through a verified clear rear.

    Rotation-only escape is intentionally not allowed inside the near-body
    envelope.  A short reverse command is returned only when both rear sensors
    are fresh, the conservative rear clearance is fully clear, and the
    dynamic tracker corroborates the threat.
    """
    if (
        not bool(dynamic_threat_confirmed)
        or not bool(rear_sensors_fresh)
        or rear_clearance_m is None
        or float(rear_clearance_m) < float(rear_slow_m)
        or float(maximum_reverse_mps) <= 0.0
    ):
        return 0.0, 0.0, False
    requested = float(requested_linear_mps)
    reverse = requested if requested < 0.0 else -float(fallback_reverse_mps)
    reverse = bound_signed_speed(
        reverse, 0.0, maximum_reverse_mps, allow_reverse=True
    )
    return reverse, float(requested_angular_radps), True


def apply_provisional_collision_intervention(
    linear_mps,
    angular_radps,
    candidates,
    maximum_angular_radps,
    preferred_turn_sign=None,
    goal_turn_sign=None,
):
    """Create a visible low-speed avoidance arc from a synchronized threat."""
    candidates = tuple(
        item
        for item in tuple(candidates or ())
        if (
            float(item.get("distance_m", float("inf"))) <= 0.85
            or (
                float(item["closest_approach_time_s"]) <= 1.80
                and float(
                    item.get(
                        "collision_clearance_m",
                        float(item["closest_approach_distance_m"])
                        - 0.60,
                    )
                )
                <= 0.25
            )
            or (
                float(item["closest_approach_time_s"]) <= 3.00
                and float(
                    item.get(
                        "collision_clearance_m",
                        float(item["closest_approach_distance_m"])
                        - 0.60,
                    )
                )
                <= 0.0
            )
            or (
                bool(item.get("mapless_dynamic_confirmed", False))
                and float(item["closest_approach_time_s"]) <= 3.60
                and float(
                    item.get(
                        "collision_clearance_m",
                        float(item["closest_approach_distance_m"])
                        - 0.60,
                    )
                )
                <= -0.10
            )
            or (
                bool(item.get("early_wide_vehicle", False))
                and float(item["closest_approach_time_s"]) <= 3.00
                and float(item["closest_approach_distance_m"]) <= 0.90
            )
        )
    )
    if not candidates:
        return float(linear_mps), float(angular_radps), None
    threat = min(
        candidates,
        key=lambda item: (
            float(item["closest_approach_time_s"]),
            float(item["closest_approach_distance_m"]),
        ),
    )
    time_to_closest = float(threat["closest_approach_time_s"])
    distance = float(threat["distance_m"])
    clearance = float(
        threat.get(
            "collision_clearance_m",
            float(threat["closest_approach_distance_m"]) - 0.60,
        )
    )
    early_wide_vehicle = bool(threat.get("early_wide_vehicle", False))
    # Start a wide-vehicle response with a shallow, visible arc.  Escalate
    # angular authority only as TTC closes; the old fixed 0.35--0.50 rad/s
    # turn produced a delayed-looking pivot and excessive lateral deviation.
    if distance <= 0.45 or (
        time_to_closest <= 0.35 and clearance <= -0.10
    ):
        forward_cap = 0.0
    elif early_wide_vehicle and time_to_closest > 2.0:
        forward_cap = 0.20
    elif early_wide_vehicle and time_to_closest > 1.0:
        forward_cap = 0.15
    elif time_to_closest <= 0.90:
        forward_cap = 0.10
    elif time_to_closest <= 1.80:
        forward_cap = 0.14
    else:
        forward_cap = 0.18
    linear = float(linear_mps)
    if linear > forward_cap:
        linear = forward_cap
    maximum_angular = abs(float(maximum_angular_radps))
    if distance <= 0.55 or time_to_closest <= 0.55:
        requested_turn_magnitude = 0.35
    elif early_wide_vehicle and time_to_closest > 2.0:
        requested_turn_magnitude = 0.16
    elif time_to_closest > 2.5:
        requested_turn_magnitude = 0.18
    elif time_to_closest > 1.5:
        requested_turn_magnitude = 0.22
    elif time_to_closest > 0.8:
        requested_turn_magnitude = 0.28
    else:
        requested_turn_magnitude = 0.32
    turn_magnitude = min(requested_turn_magnitude, maximum_angular)
    lateral = float(threat.get("lateral_m", 0.0))
    # Positive lateral is left of the robot, so turn right, and vice versa.
    # A recent verified escape direction is authoritative while the robot is
    # still inside the obstacle envelope.  Tracker identity/lateral estimates
    # can jump during an in-place turn and must not reverse that escape.
    if preferred_turn_sign is not None:
        turn_sign = -1.0 if float(preferred_turn_sign) < 0.0 else 1.0
    elif lateral >= 0.30:
        turn_sign = -1.0 if lateral >= 0.0 else 1.0
    elif lateral <= -0.30:
        turn_sign = 1.0
    elif goal_turn_sign is not None:
        turn_sign = -1.0 if float(goal_turn_sign) < 0.0 else 1.0
    else:
        turn_sign = -1.0 if lateral >= 0.0 else 1.0
    angular = turn_sign * turn_magnitude
    angular = clamp(angular, -maximum_angular, maximum_angular)
    selected = dict(threat)
    selected["collision_clearance_m"] = clearance
    selected["forward_cap_mps"] = forward_cap
    selected["turn_magnitude_radps"] = turn_magnitude
    selected["turn_sign"] = turn_sign
    return linear, angular, selected


def stabilize_goal_heading(
    linear_mps,
    angular_radps,
    heading_error_rad,
    maximum_angular_radps,
    activation_error_rad=0.25,
):
    """Reject a nominal turn that increases a large point-goal error."""
    linear = float(linear_mps)
    angular = float(angular_radps)
    error = float(heading_error_rad)
    maximum_angular = abs(float(maximum_angular_radps))
    if (
        maximum_angular <= 0.0
        or abs(error) < float(activation_error_rad)
        or angular * error > 0.0
    ):
        return linear, angular, False
    correction = clamp(
        1.2 * error,
        -maximum_angular,
        maximum_angular,
    )
    return min(linear, 0.20), correction, True


def apply_reactive_arc_hold(
    linear_mps,
    angular_radps,
    turn_sign,
    maximum_angular_radps,
    maximum_hold_linear_mps=0.18,
    minimum_hold_angular_radps=0.30,
):
    """Continue a positive-speed avoidance arc across short tracker gaps.

    The hold never creates forward motion from a stop or reverse command.  It
    only limits an already-positive command and preserves the last avoidance
    direction, so higher-priority guard stops remain fail-closed.
    """
    linear = float(linear_mps)
    angular = float(angular_radps)
    if turn_sign is None or linear <= 0.0:
        return linear, angular, False
    maximum_angular = abs(float(maximum_angular_radps))
    if maximum_angular <= 0.0:
        return linear, angular, False
    sign = -1.0 if float(turn_sign) < 0.0 else 1.0
    linear = min(linear, max(float(maximum_hold_linear_mps), 0.0))
    turn_magnitude = min(
        max(float(minimum_hold_angular_radps), 0.0),
        maximum_angular,
    )
    angular = sign * max(abs(angular), turn_magnitude)
    angular = clamp(angular, -maximum_angular, maximum_angular)
    return linear, angular, True


def apply_turn_direction_lock(
    linear_mps,
    angular_radps,
    turn_sign,
    maximum_angular_radps,
    minimum_angular_radps=0.30,
):
    """Preserve a verified turn sign without creating linear motion."""
    linear = max(float(linear_mps), 0.0)
    if turn_sign is None:
        return linear, float(angular_radps), False
    maximum = abs(float(maximum_angular_radps))
    if maximum <= 0.0:
        return linear, float(angular_radps), False
    sign = -1.0 if float(turn_sign) < 0.0 else 1.0
    magnitude = min(
        max(abs(float(angular_radps)), float(minimum_angular_radps)),
        maximum,
    )
    return linear, sign * magnitude, True


def limit_angular_command_step(
    angular_radps,
    previous_angular_radps,
    maximum_step_radps=0.35,
):
    """Rate-limit planner sign reversals before the physical smoother."""
    requested = float(angular_radps)
    previous = float(previous_angular_radps)
    step = abs(float(maximum_step_radps))
    limited = clamp(requested, previous - step, previous + step)
    return limited, abs(limited - requested) > 1.0e-12


def angular_slew_step_budget(
    elapsed_s,
    maximum_slew_radps2,
    maximum_elapsed_s,
):
    """Return a bounded angular-command step from a real control interval.

    A fixed per-cycle step silently becomes a larger physical angular
    acceleration whenever the control period is shorter than expected.  The
    elapsed interval is therefore used to form the step budget.  Conversely,
    a delayed cycle must not earn an arbitrarily large one-shot turn reversal,
    so the interval is capped at the nominal control period.
    """
    elapsed = float(elapsed_s)
    slew = float(maximum_slew_radps2)
    maximum_elapsed = float(maximum_elapsed_s)
    if slew <= 0.0:
        raise ValueError("maximum angular slew must be positive")
    if maximum_elapsed <= 0.0:
        raise ValueError("maximum elapsed interval must be positive")
    return slew * min(max(elapsed, 0.0), maximum_elapsed)


def apply_front_clearance_arc_governor(
    linear_mps,
    angular_radps,
    front_clearance_m,
    maximum_angular_radps,
    front_obstacle_angle_rad=None,
    activation_clearance_m=1.60,
    hard_stop_clearance_m=0.80,
    maximum_arc_linear_mps=0.20,
    minimum_arc_linear_mps=0.10,
    minimum_arc_angular_radps=0.30,
):
    """Tighten an existing planner turn before the front watchdog stops it.

    This governor never invents an avoidance direction and never labels an
    object dynamic.  It only preserves a turn already selected by the planner,
    reducing its radius as a static or unknown obstacle enters the forward
    clearance envelope.  The robot gateway remains authoritative at and below
    its hard-stop distance.
    """
    linear = float(linear_mps)
    angular = float(angular_radps)
    if front_clearance_m is None or linear < 0.0:
        return linear, angular, None
    clearance = float(front_clearance_m)
    activation = float(activation_clearance_m)
    hard_stop = float(hard_stop_clearance_m)
    maximum_angular = abs(float(maximum_angular_radps))
    if (
        maximum_angular <= 0.0
        or clearance >= activation
        or (
            abs(angular) < 0.20
            and front_obstacle_angle_rad is None
        )
        or (
            linear == 0.0
            and front_obstacle_angle_rad is None
        )
    ):
        return linear, angular, None

    span = max(activation - hard_stop, 1.0e-9)
    proximity = clamp((activation - clearance) / span, 0.0, 1.0)
    linear_cap = (
        float(maximum_arc_linear_mps)
        + proximity
        * (
            float(minimum_arc_linear_mps)
            - float(maximum_arc_linear_mps)
        )
    )
    linear = min(linear, max(linear_cap, 0.0))
    turn_sign = -1.0 if angular < 0.0 else 1.0
    obstacle_angle = (
        None
        if front_obstacle_angle_rad is None
        else float(front_obstacle_angle_rad)
    )
    turn_away_applied = False
    if obstacle_angle is not None and abs(obstacle_angle) >= 0.05:
        turn_sign = -1.0 if obstacle_angle > 0.0 else 1.0
        turn_away_applied = angular * turn_sign <= 0.0
    turn_floor = min(
        max(float(minimum_arc_angular_radps), 0.0),
        maximum_angular,
    )
    turn_magnitude = max(abs(angular), turn_floor)
    if obstacle_angle is not None:
        turn_magnitude = min(turn_magnitude, min(0.35, maximum_angular))
    angular = turn_sign * turn_magnitude
    angular = clamp(angular, -maximum_angular, maximum_angular)
    return linear, angular, {
        "front_clearance_m": clearance,
        "linear_cap_mps": linear_cap,
        "turn_sign": turn_sign,
        "proximity": proximity,
        "front_obstacle_angle_rad": obstacle_angle,
        "turn_away_applied": turn_away_applied,
    }


def apply_near_body_speed_governor(
    linear_mps,
    near_body_clearance_m,
    activation_clearance_m=1.20,
    stop_clearance_m=0.35,
    maximum_linear_mps=0.18,
    minimum_linear_mps=0.03,
):
    """Slow before a side obstacle rotates into the forward scan sector."""
    linear = float(linear_mps)
    if near_body_clearance_m is None or linear <= 0.0:
        return linear, None
    clearance = float(near_body_clearance_m)
    activation = float(activation_clearance_m)
    stop = float(stop_clearance_m)
    if clearance >= activation:
        return linear, None
    proximity = clamp(
        (activation - clearance) / max(activation - stop, 1.0e-9),
        0.0,
        1.0,
    )
    linear_cap = (
        float(maximum_linear_mps)
        + proximity
        * (float(minimum_linear_mps) - float(maximum_linear_mps))
    )
    return min(linear, max(linear_cap, 0.0)), {
        "near_body_clearance_m": clearance,
        "linear_cap_mps": linear_cap,
        "proximity": proximity,
    }


def verified_near_body_turn_escape(
    requested_angular_radps,
    nearest_obstacle_angle_rad,
    left_clearance_m,
    right_clearance_m,
    maximum_angular_radps,
    required_side_clearance_m=0.55,
    escape_angular_radps=0.18,
):
    """Allow a zero-linear escape turn only toward a lidar-verified open side."""
    requested = float(requested_angular_radps)
    maximum = abs(float(maximum_angular_radps))
    if maximum <= 0.0 or abs(requested) <= 1.0e-9:
        return 0.0, None
    angle = float(nearest_obstacle_angle_rad)
    if angle > 0.12:
        turn_sign = -1.0
    elif angle < -0.12:
        turn_sign = 1.0
    else:
        turn_sign = -1.0 if requested < 0.0 else 1.0
    side_clearance = (
        left_clearance_m if turn_sign > 0.0 else right_clearance_m
    )
    if (
        side_clearance is None
        or float(side_clearance) < float(required_side_clearance_m)
        or requested * turn_sign <= 0.0
    ):
        return 0.0, None
    magnitude = min(abs(float(escape_angular_radps)), maximum)
    return turn_sign * magnitude, {
        "nearest_obstacle_angle_rad": angle,
        "side_clearance_m": float(side_clearance),
        "turn_sign": turn_sign,
        "angular_radps": magnitude,
    }


def apply_front_arc_creep(
    value,
    front_clearance_m,
    front_stop_m,
    near_body_stop_m,
    maximum_creep_mps=0.08,
):
    """Replace a forward hard stop with a vanishing low-speed arc envelope."""
    value = float(value)
    if value <= 0.0 or float(front_clearance_m) > float(front_stop_m):
        return value, None
    scale = (
        float(front_clearance_m) - float(near_body_stop_m)
    ) / max(
        float(front_stop_m) - float(near_body_stop_m),
        1.0e-9,
    )
    capped = min(
        value,
        float(maximum_creep_mps) * clamp(scale, 0.0, 1.0),
    )
    return capped, "front_arc_creep"


def motion_session_expired(session_started_at, now, timeout_s):
    """A live motion lease does not age until the first command arrives."""
    if session_started_at is None:
        return False
    return float(now) - float(session_started_at) > float(timeout_s)


def apply_directional_clearance(
    value,
    front_clearance_m,
    rear_clearance_m,
    front_stop_m,
    front_slow_m,
    rear_stop_m,
    rear_slow_m,
):
    value = float(value)
    if value > 0.0 and front_clearance_m <= float(front_stop_m):
        return 0.0, "front_hard_stop"
    if value > 0.0 and front_clearance_m < float(front_slow_m):
        scale = (
            float(front_clearance_m) - float(front_stop_m)
        ) / max(float(front_slow_m) - float(front_stop_m), 1.0e-9)
        return value * clamp(scale, 0.0, 1.0), "front_slow"
    if value < 0.0 and rear_clearance_m <= float(rear_stop_m):
        return 0.0, "rear_hard_stop"
    if value < 0.0 and rear_clearance_m < float(rear_slow_m):
        scale = (
            float(rear_clearance_m) - float(rear_stop_m)
        ) / max(float(rear_slow_m) - float(rear_stop_m), 1.0e-9)
        return value * clamp(scale, 0.0, 1.0), "rear_slow"
    return value, "command_accepted"


def paired_rear_clearance(right_clearance_m, left_clearance_m):
    """Return fail-closed two-sensor rear clearance."""
    if right_clearance_m is None or left_clearance_m is None:
        return None
    return min(float(right_clearance_m), float(left_clearance_m))


def paired_rear_is_fresh(
    right_clearance_m,
    left_clearance_m,
    right_age_s,
    left_age_s,
    watchdog_s,
):
    if paired_rear_clearance(right_clearance_m, left_clearance_m) is None:
        return False
    watchdog_s = float(watchdog_s)
    return (
        0.0 <= float(right_age_s) <= watchdog_s
        and 0.0 <= float(left_age_s) <= watchdog_s
    )
