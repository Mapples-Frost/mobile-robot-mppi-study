# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import io
import os

try:
    import yaml
except ImportError:
    yaml = None

THIS_FILE = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "mppi_hardware_bridge",
    "config",
    "lab_runtime.yaml",
)

DEFAULT_MPPI_SCAN_OBSTACLE_CONFIG = {
    "horizon": 25,
    "num_samples": 350,
    "realtime_degraded_num_samples": 200,
    "realtime_degraded_horizon": 20,
    "emergency_degraded_num_samples": 180,
    "emergency_degraded_horizon": 20,
    "realtime_core_ms_threshold": 180.0,
    "realtime_recover_core_ms": 140.0,
    "emergency_degraded_core_streak": 4,
    "realtime_recover_streak": 5,
    "temperature": 14.0,
    "omega_std": 0.95,
    "v_std": 0.16,
    "scan_obstacle_mode": "geometric",
    "scan_downsample_step": 1,
    "dynamic_obstacle_max_count": 4,
    "geometric_max_obstacle_count": 4,
    "geometric_min_segment_points": 4,
    "geometric_line_spacing": 0.20,
    "geometric_obstacle_inflation": 0.04,
    "geometric_smooth_window_size": 3,
    "geometric_max_neighbor_distance": 0.15,
    "geometric_tangent_jump_threshold_rad": 0.55,
    "geometric_curvature_threshold": 2.5,
    "geometric_curvature_delta_threshold": 2.0,
    "geometric_line_error_threshold": 0.035,
    "geometric_line_curvature_threshold": 0.35,
    "geometric_min_line_length": 0.15,
    "geometric_circle_error_threshold": 0.035,
    "geometric_circle_curvature_std_threshold": 0.60,
    "geometric_min_circle_radius": 0.04,
    "geometric_max_circle_radius": 0.60,
    "geometric_min_circle_arc_angle": 0.35,
    "geometric_line_sample_spacing": 0.20,
    "geometric_line_max_obstacles": 3,
    "geometric_edge_corner_radius_scale": 1.5,
    "geometric_irregular_max_points": 3,
}

DEFAULT_LIMITS_STABILITY_CONFIG = {
    "omega_slew_rate": 0.14,
    "omega_lowpass_alpha": 0.03,
    "clear_goal_tracking_alpha": 0.35,
    "clear_goal_tracking_min_omega": 0.05,
    "obstacle_omega_slew_rate": 2.5,
    "obstacle_omega_lowpass_alpha": 0.85,
    "hard_stop_omega_slew_rate": 4.0,
    "hard_stop_omega_lowpass_alpha": 1.0,
}

DEFAULT_SAFETY_STABILITY_CONFIG = {
    "hard_stop_distance": 0.18,
    "hard_stop_release_margin": 0.04,
    "front_soft_block_distance": 0.28,
    "rotate_escape_w": 1.2,
    "avoidance_side_hold_sec": 1.2,
    "scan_angle_offset_deg": 0.0,
    "base_frame_id": "base_footprint",
    "use_scan_tf": True,
    "side_stop_distance": 0.0,
    "side_angle_deg": 120.0,
    "side_soft_distance": 0.26,
    "side_hard_distance": 0.18,
    "side_release_distance": 0.34,
    "side_soft_min_v": 0.05,
    "side_avoid_v_scale": 0.80,
    "side_avoid_min_omega": 0.10,
    "side_avoid_max_omega": 0.18,
    "side_avoid_hold_sec": 0.8,
    "side_front_exclusion_angle_deg": 25.0,
    "near_body_stop_radius": 0.0,
    "front_turn_distance": 0.85,
    "front_turn_release_distance": 0.95,
    "front_slow_min_scale": 0.45,
    "soft_avoid_min_v": 0.055,
    "soft_avoid_target_v": 0.075,
    "front_corridor_half_width": 0.38,
    "front_turn_min_omega": 0.12,
    "front_turn_max_omega": 0.22,
    "front_turn_keep_v_scale": 0.75,
    "obstacle_turn_hold_sec": 0.7,
    "obstacle_turn_damp_yaw_error_deg": 14.0,
    "obstacle_turn_max_yaw_error_deg": 22.0,
    "obstacle_turn_bias_omega": 0.06,
    "obstacle_turn_override_min_risk": 0.65,
    "obstacle_turn_score_deadband": 0.20,
    "obstacle_turn_lock_min_sec": 0.25,
    "obstacle_turn_lock_max_sec": 0.7,
    "mppi_omega_trust_frames": 3,
    "hard_override_margin_streak_frames": 3,
    "hard_override_conflict_streak_frames": 2,
    "corridor_switch_margin": 0.15,
    "corridor_switch_streak_frames": 3,
    "corridor_side_risk_weight": 1.5,
    "corridor_goal_error_weight": 0.8,
    "corridor_clearance_weight": 1.0,
    "avoidance_release_clear_frames": 3,
    "avoidance_progress_release_frames": 3,
    "avoidance_no_obstacle_release_frames": 2,
    "allow_rotate_in_front_stop": True,
    "front_stop_recovery_w_max": 0.25,
    "front_stop_recovery_w_min": 0.15,
    "hard_stop_recovery_no_improve_frames": 8,
    "hard_stop_recovery_switch_min_sec": 0.8,
    "front_stop_release_distance": 0.31,
    "front_stop_recovery_use_goal_direction": True,
    "planner_creep_enable": True,
    "planner_creep_v": 0.060,
    "planner_creep_max_omega": 0.14,
    "creep_max_duration_sec": 2.0,
}

DEFAULT_TRACKING_CONFIG = {
    "enable_goal_tracking_override": True,
    "yaw_align_deg": 115.0,
    "yaw_slow_deg": 55.0,
    "yaw_deadband_deg": 3.0,
    "k_yaw": 0.18,
    "omega_align_max": 0.18,
    "omega_track_max": 0.18,
    "omega_deadband_max": 0.03,
    "min_v_scale": 0.25,
    "suppress_wrong_sign_omega": False,
    "reset_smoother_on_omega_sign_flip": False,
}


class ConfigNode(dict):
    """
    一个很轻量的配置对象。

    普通 dict 只能这样访问：
        cfg["goal"]["x"]

    ConfigNode 允许这样访问：
        cfg.goal.x

    这样后面写 adapter 会清楚很多。
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def to_config_node(obj):
    """
    把 yaml 读出来的 dict/list 递归转换成 ConfigNode。
    """
    if isinstance(obj, dict):
        node = ConfigNode()
        for key, value in obj.items():
            node[key] = to_config_node(value)
        return node

    if isinstance(obj, list):
        return [to_config_node(item) for item in obj]

    return obj


def require_keys(section, required_keys, section_name):
    """
    检查某个配置块里必须有指定字段。

    section:
        某一块配置，比如 raw["goal"]

    required_keys:
        必须存在的 key 列表，比如 ["x", "y", "tolerance"]

    section_name:
        用于报错时说明是哪一块配置缺字段。
    """
    missing = []

    for key in required_keys:
        if key not in section:
            missing.append(key)

    if missing:
        raise ValueError(
            "Missing keys in section '{}': {}".format(
                section_name,
                missing,
            )
        )


def apply_optional_defaults(raw):
    mppi = raw.get("mppi", {})
    for key, value in DEFAULT_MPPI_SCAN_OBSTACLE_CONFIG.items():
        if key not in mppi:
            mppi[key] = value

    limits = raw.get("limits", {})
    for key, value in DEFAULT_LIMITS_STABILITY_CONFIG.items():
        if key not in limits:
            limits[key] = value

    safety = raw.get("safety", {})
    for key, value in DEFAULT_SAFETY_STABILITY_CONFIG.items():
        if key not in safety:
            safety[key] = value

    if "tracking" not in raw:
        raw["tracking"] = {}
    tracking = raw.get("tracking", {})
    for key, value in DEFAULT_TRACKING_CONFIG.items():
        if key not in tracking:
            tracking[key] = value


def strip_yaml_comment(line):
    in_single_quote = False
    in_double_quote = False

    for index, char in enumerate(line):
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            return line[:index]

    return line


def parse_simple_yaml_scalar(value):
    value = value.strip()

    if not value:
        return ""

    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]

    lower_value = value.lower()
    if lower_value in ("true", "false"):
        return lower_value == "true"
    if lower_value in ("null", "none", "~"):
        return None

    try:
        if (
            "." not in value
            and "e" not in lower_value
            and value.lstrip("-").isdigit()
        ):
            return int(value)
        return float(value)
    except ValueError:
        return value


def simple_yaml_load(text):
    """
    Tiny fallback parser for the nested key/value YAML used by lab_runtime.yaml.

    ROS Kinetic normally provides PyYAML. This keeps field tests usable if a
    stripped Python2 environment lacks the yaml package.
    """
    root = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        if "\t" in raw_line:
            raise ValueError("Tabs are not supported in lab_runtime.yaml.")

        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        content = strip_yaml_comment(raw_line[indent:]).strip()
        if not content:
            continue

        if ":" not in content:
            raise ValueError("Unsupported YAML line: {}".format(raw_line))

        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_simple_yaml_scalar(value)

    return root


def validate_lab_runtime_config(raw):
    """
    对 lab_runtime.yaml 做最基本的安全检查。

    这一步不是为了复杂，而是为了避免现场因为少写字段、
    拼错字段、速度参数过大而直接出危险。
    """

    require_keys(
        raw,
        [
            "mode",
            "frame",
            "ros",
            "goal",
            "boundary",
            "limits",
            "safety",
            "mppi",
            "logging",
        ],
        "root",
    )

    apply_optional_defaults(raw)

    require_keys(
        raw["frame"],
        [
            "type",
            "origin",
            "x_axis",
            "y_axis",
            "position_unit",
            "linear_velocity_unit",
            "angular_velocity_unit",
            "config_angle_unit",
        ],
        "frame",
    )

    require_keys(
        raw["ros"],
        [
            "odom_topic",
            "fallback_odom_topic",
            "scan_topic",
            "cmd_vel_topic",
            "runtime_goal_topic",
        ],
        "ros",
    )

    require_keys(
        raw["goal"],
        [
            "x",
            "y",
            "tolerance",
        ],
        "goal",
    )

    require_keys(
        raw["boundary"],
        [
            "enabled",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
        ],
        "boundary",
    )

    require_keys(
        raw["limits"],
        [
            "profile",
            "v_max",
            "w_max",
            "k_yaw",
            "yaw_stop_deg",
        ],
        "limits",
    )

    require_keys(
        raw["safety"],
        [
            "enable_publish",
            "enable_scan_guard",
            "enable_dynamic_obstacle_cost",
            "emergency_stop",
            "front_stop_distance",
            "front_slow_distance",
            "front_angle_deg",
            "local_obstacle_radius",
        ],
        "safety",
    )

    require_keys(
        raw["mppi"],
        [
            "use_existing_planner",
            "use_anisotropic_sampling",
            "sdf_influence_distance",
            "scan_obstacle_influence_distance",
        ],
        "mppi",
    )

    require_keys(
        raw["logging"],
        [
            "print_debug",
            "save_runtime_log",
        ],
        "logging",
    )

    if raw["frame"]["type"] != "relative_to_start":
        raise ValueError(
            "For the first hardware version, frame.type must be 'relative_to_start'."
        )

    if raw["limits"]["v_max"] <= 0.0:
        raise ValueError("limits.v_max must be positive.")

    if raw["limits"]["w_max"] <= 0.0:
        raise ValueError("limits.w_max must be positive.")

    if raw["limits"]["v_max"] > 0.50:
        raise ValueError(
            "limits.v_max is too large for this hardware bridge profile. "
            "Please keep it <= 0.50 m/s."
        )

    if raw["limits"]["w_max"] > 4.00:
        raise ValueError(
            "limits.w_max is too large for this hardware bridge profile. "
            "Please keep it <= 4.00 rad/s."
        )

    if raw["limits"]["omega_slew_rate"] <= 0.0:
        raise ValueError("limits.omega_slew_rate must be positive.")

    if (
        raw["limits"]["omega_lowpass_alpha"] < 0.0
        or raw["limits"]["omega_lowpass_alpha"] > 1.0
    ):
        raise ValueError("limits.omega_lowpass_alpha must be in [0, 1].")

    if raw["goal"]["tolerance"] <= 0.0:
        raise ValueError("goal.tolerance must be positive.")

    if raw["boundary"]["xmin"] >= raw["boundary"]["xmax"]:
        raise ValueError("boundary.xmin must be smaller than boundary.xmax.")

    if raw["boundary"]["ymin"] >= raw["boundary"]["ymax"]:
        raise ValueError("boundary.ymin must be smaller than boundary.ymax.")

    if raw["safety"]["front_stop_distance"] <= 0.0:
        raise ValueError("safety.front_stop_distance must be positive.")

    if raw["safety"]["hard_stop_distance"] <= 0.0:
        raise ValueError("safety.hard_stop_distance must be positive.")

    if raw["safety"]["hard_stop_distance"] > raw["safety"]["front_stop_distance"]:
        raise ValueError(
            "safety.hard_stop_distance should be <= front_stop_distance."
        )

    if raw["safety"]["front_soft_block_distance"] < raw["safety"]["front_stop_distance"]:
        raise ValueError(
            "safety.front_soft_block_distance should be >= front_stop_distance."
        )

    if raw["safety"]["front_slow_distance"] < raw["safety"]["front_soft_block_distance"]:
        raise ValueError(
            "safety.front_slow_distance should be >= front_soft_block_distance."
        )

    if (
        raw["safety"]["front_slow_min_scale"] < 0.0
        or raw["safety"]["front_slow_min_scale"] > 1.0
    ):
        raise ValueError("safety.front_slow_min_scale must be in [0, 1].")

    if raw["safety"]["soft_avoid_min_v"] < 0.0:
        raise ValueError("safety.soft_avoid_min_v must be >= 0.")

    if raw["safety"]["soft_avoid_target_v"] < raw["safety"]["soft_avoid_min_v"]:
        raise ValueError("safety.soft_avoid_target_v must be >= soft_avoid_min_v.")

    if raw["safety"]["rotate_escape_w"] <= 0.0:
        raise ValueError("safety.rotate_escape_w must be positive.")

    if raw["safety"]["avoidance_side_hold_sec"] < 0.0:
        raise ValueError("safety.avoidance_side_hold_sec must be >= 0.")

    if raw["safety"]["side_stop_distance"] < 0.0:
        raise ValueError("safety.side_stop_distance must be >= 0.")

    if raw["safety"]["side_angle_deg"] <= 0.0:
        raise ValueError("safety.side_angle_deg must be positive.")

    if raw["safety"]["side_hard_distance"] < 0.0:
        raise ValueError("safety.side_hard_distance must be >= 0.")

    if raw["safety"]["side_soft_distance"] < raw["safety"]["side_hard_distance"]:
        raise ValueError(
            "safety.side_soft_distance should be >= side_hard_distance."
        )

    if raw["safety"]["side_release_distance"] < raw["safety"]["side_soft_distance"]:
        raise ValueError(
            "safety.side_release_distance should be >= side_soft_distance."
        )

    if (
        raw["safety"]["side_avoid_v_scale"] < 0.0
        or raw["safety"]["side_avoid_v_scale"] > 1.0
    ):
        raise ValueError("safety.side_avoid_v_scale must be in [0, 1].")

    if raw["safety"]["side_soft_min_v"] < 0.0:
        raise ValueError("safety.side_soft_min_v must be >= 0.")

    if raw["safety"]["side_avoid_min_omega"] < 0.0:
        raise ValueError("safety.side_avoid_min_omega must be >= 0.")

    if raw["safety"]["side_avoid_max_omega"] < raw["safety"]["side_avoid_min_omega"]:
        raise ValueError(
            "safety.side_avoid_max_omega must be >= side_avoid_min_omega."
        )

    if raw["safety"]["side_avoid_hold_sec"] < 0.0:
        raise ValueError("safety.side_avoid_hold_sec must be >= 0.")

    if raw["safety"]["side_front_exclusion_angle_deg"] < 0.0:
        raise ValueError("safety.side_front_exclusion_angle_deg must be >= 0.")

    if raw["safety"]["side_front_exclusion_angle_deg"] >= raw["safety"]["side_angle_deg"]:
        raise ValueError(
            "safety.side_front_exclusion_angle_deg should be < side_angle_deg."
        )

    if raw["safety"]["near_body_stop_radius"] < 0.0:
        raise ValueError("safety.near_body_stop_radius must be >= 0.")

    if raw["safety"]["front_turn_distance"] < raw["safety"]["front_slow_distance"]:
        raise ValueError(
            "safety.front_turn_distance should be >= front_slow_distance."
        )

    if raw["safety"]["front_turn_release_distance"] < raw["safety"]["front_turn_distance"]:
        raise ValueError(
            "safety.front_turn_release_distance should be >= front_turn_distance."
        )

    if raw["safety"]["front_corridor_half_width"] <= 0.0:
        raise ValueError("safety.front_corridor_half_width must be positive.")

    if raw["safety"]["front_turn_min_omega"] < 0.0:
        raise ValueError("safety.front_turn_min_omega must be >= 0.")

    if raw["safety"]["front_turn_max_omega"] < raw["safety"]["front_turn_min_omega"]:
        raise ValueError(
            "safety.front_turn_max_omega must be >= front_turn_min_omega."
        )

    if (
        raw["safety"]["front_turn_keep_v_scale"] < 0.0
        or raw["safety"]["front_turn_keep_v_scale"] > 1.0
    ):
        raise ValueError("safety.front_turn_keep_v_scale must be in [0, 1].")

    if raw["safety"]["obstacle_turn_hold_sec"] < 0.0:
        raise ValueError("safety.obstacle_turn_hold_sec must be >= 0.")

    if raw["safety"]["obstacle_turn_bias_omega"] < 0.0:
        raise ValueError("safety.obstacle_turn_bias_omega must be >= 0.")

    if (
        raw["safety"]["obstacle_turn_override_min_risk"] < 0.0
        or raw["safety"]["obstacle_turn_override_min_risk"] > 1.0
    ):
        raise ValueError("safety.obstacle_turn_override_min_risk must be in [0, 1].")

    if raw["safety"]["obstacle_turn_score_deadband"] < 0.0:
        raise ValueError("safety.obstacle_turn_score_deadband must be >= 0.")

    if raw["safety"]["obstacle_turn_lock_min_sec"] < 0.0:
        raise ValueError("safety.obstacle_turn_lock_min_sec must be >= 0.")

    if (
        raw["safety"]["obstacle_turn_lock_max_sec"]
        < raw["safety"]["obstacle_turn_lock_min_sec"]
    ):
        raise ValueError(
            "safety.obstacle_turn_lock_max_sec must be >= obstacle_turn_lock_min_sec."
        )

    if int(raw["safety"]["mppi_omega_trust_frames"]) < 0:
        raise ValueError("safety.mppi_omega_trust_frames must be >= 0.")

    if raw["safety"]["obstacle_turn_damp_yaw_error_deg"] < 0.0:
        raise ValueError("safety.obstacle_turn_damp_yaw_error_deg must be >= 0.")

    if (
        raw["safety"]["obstacle_turn_max_yaw_error_deg"]
        < raw["safety"]["obstacle_turn_damp_yaw_error_deg"]
    ):
        raise ValueError(
            "safety.obstacle_turn_max_yaw_error_deg must be >= "
            "obstacle_turn_damp_yaw_error_deg."
        )

    corridor_score_keys = [
        "corridor_switch_margin",
        "corridor_side_risk_weight",
        "corridor_goal_error_weight",
        "corridor_clearance_weight",
    ]
    for key in corridor_score_keys:
        if raw["safety"][key] < 0.0:
            raise ValueError("safety.{} must be >= 0.".format(key))

    if raw["safety"]["front_stop_recovery_w_max"] < 0.0:
        raise ValueError("safety.front_stop_recovery_w_max must be >= 0.")

    if raw["safety"]["front_stop_recovery_w_min"] < 0.0:
        raise ValueError("safety.front_stop_recovery_w_min must be >= 0.")

    if (
        raw["safety"]["front_stop_recovery_w_min"]
        > raw["safety"]["front_stop_recovery_w_max"]
    ):
        raise ValueError(
            "safety.front_stop_recovery_w_min must be <= front_stop_recovery_w_max."
        )

    if raw["safety"]["front_stop_release_distance"] < raw["safety"]["front_stop_distance"]:
        raise ValueError(
            "safety.front_stop_release_distance should be >= front_stop_distance."
        )

    if raw["safety"]["planner_creep_v"] < 0.0:
        raise ValueError("safety.planner_creep_v must be >= 0.")

    if raw["safety"]["planner_creep_max_omega"] < 0.0:
        raise ValueError("safety.planner_creep_max_omega must be >= 0.")

    if raw["safety"]["creep_max_duration_sec"] < 0.0:
        raise ValueError("safety.creep_max_duration_sec must be >= 0.")

    tracking = raw["tracking"]
    positive_tracking_keys = [
        "yaw_deadband_deg",
        "yaw_slow_deg",
        "yaw_align_deg",
        "k_yaw",
        "omega_track_max",
        "omega_align_max",
        "omega_deadband_max",
    ]
    for key in positive_tracking_keys:
        if float(tracking[key]) <= 0.0:
            raise ValueError("tracking.{} must be positive.".format(key))

    if float(tracking["yaw_deadband_deg"]) > float(tracking["yaw_slow_deg"]):
        raise ValueError("tracking.yaw_deadband_deg must be <= yaw_slow_deg.")
    if float(tracking["yaw_slow_deg"]) > float(tracking["yaw_align_deg"]):
        raise ValueError("tracking.yaw_slow_deg must be <= yaw_align_deg.")
    if float(tracking["min_v_scale"]) < 0.0 or float(tracking["min_v_scale"]) > 1.0:
        raise ValueError("tracking.min_v_scale must be in [0, 1].")

    scan_obstacle_mode = str(raw["mppi"]["scan_obstacle_mode"]).strip().lower()
    if scan_obstacle_mode not in ("raw", "geometric"):
        raise ValueError("mppi.scan_obstacle_mode must be 'raw' or 'geometric'.")
    raw["mppi"]["scan_obstacle_mode"] = scan_obstacle_mode

    if int(raw["mppi"]["scan_downsample_step"]) <= 0:
        raise ValueError("mppi.scan_downsample_step must be positive.")

    if int(raw["mppi"]["dynamic_obstacle_max_count"]) <= 0:
        raise ValueError("mppi.dynamic_obstacle_max_count must be positive.")

    if int(raw["mppi"]["geometric_max_obstacle_count"]) <= 0:
        raise ValueError("mppi.geometric_max_obstacle_count must be positive.")

    if int(raw["mppi"]["geometric_min_segment_points"]) <= 0:
        raise ValueError("mppi.geometric_min_segment_points must be positive.")

    numeric_positive_mppi_keys = [
        "horizon",
        "num_samples",
        "temperature",
        "omega_std",
        "v_std",
        "geometric_line_spacing",
        "geometric_smooth_window_size",
        "geometric_max_neighbor_distance",
        "geometric_tangent_jump_threshold_rad",
        "geometric_curvature_threshold",
        "geometric_curvature_delta_threshold",
        "geometric_line_error_threshold",
        "geometric_min_line_length",
        "geometric_circle_error_threshold",
        "geometric_circle_curvature_std_threshold",
        "geometric_min_circle_radius",
        "geometric_max_circle_radius",
        "geometric_min_circle_arc_angle",
        "geometric_line_sample_spacing",
        "geometric_line_max_obstacles",
        "geometric_edge_corner_radius_scale",
        "geometric_irregular_max_points",
    ]
    for key in numeric_positive_mppi_keys:
        if float(raw["mppi"][key]) <= 0.0:
            raise ValueError("mppi.{} must be positive.".format(key))

    if float(raw["mppi"]["geometric_line_curvature_threshold"]) < 0.0:
        raise ValueError("mppi.geometric_line_curvature_threshold must be >= 0.")

    if float(raw["mppi"]["geometric_obstacle_inflation"]) < 0.0:
        raise ValueError("mppi.geometric_obstacle_inflation must be >= 0.")

    if (
        float(raw["mppi"]["geometric_min_circle_radius"])
        > float(raw["mppi"]["geometric_max_circle_radius"])
    ):
        raise ValueError(
            "mppi.geometric_min_circle_radius must be <= geometric_max_circle_radius."
        )


def load_lab_runtime_config(config_path):
    """
    读取五一实验现场配置文件。

    config_path:
        lab_runtime.yaml 的路径。

    返回：
        ConfigNode 对象，后面可以这样用：
            cfg.goal.x
            cfg.limits.v_max
            cfg.safety.enable_publish
    """
    if not os.path.exists(config_path):
        raise IOError("Config file not found: {}".format(config_path))

    with io.open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read()

    if yaml is None:
        raw = simple_yaml_load(config_text)
    else:
        raw = yaml.safe_load(config_text)

    if raw is None:
        raise ValueError("Config file is empty: {}".format(config_path))

    validate_lab_runtime_config(raw)

    return to_config_node(raw)


def print_config_summary(cfg):
    """
    打印一份简短摘要，方便你确认配置有没有读错。
    """
    print("")
    print("Loaded lab runtime config")
    print("-------------------------")
    print("mode: {}".format(cfg.mode))
    print("frame.type: {}".format(cfg.frame.type))
    print("frame.origin: {}".format(cfg.frame.origin))
    print("frame.x_axis: {}".format(cfg.frame.x_axis))
    print("")
    print("ROS topics:")
    print("  odom: {}".format(cfg.ros.odom_topic))
    print("  fallback odom: {}".format(cfg.ros.fallback_odom_topic))
    print("  scan: {}".format(cfg.ros.scan_topic))
    print("  cmd_vel: {}".format(cfg.ros.cmd_vel_topic))
    print("  runtime goal: {}".format(cfg.ros.runtime_goal_topic))
    print("")
    print("Goal in experiment frame:")
    print("  x = {:.3f} m".format(cfg.goal.x))
    print("  y = {:.3f} m".format(cfg.goal.y))
    print("  tolerance = {:.3f} m".format(cfg.goal.tolerance))
    print("")
    print("Boundary in experiment frame:")
    print("  x: [{:.3f}, {:.3f}] m".format(cfg.boundary.xmin, cfg.boundary.xmax))
    print("  y: [{:.3f}, {:.3f}] m".format(cfg.boundary.ymin, cfg.boundary.ymax))
    print("")
    print("Limits:")
    print("  profile: {}".format(cfg.limits.profile))
    print("  v_max = {:.3f} m/s".format(cfg.limits.v_max))
    print("  w_max = {:.3f} rad/s".format(cfg.limits.w_max))
    print("  yaw_stop_deg = {:.1f} deg".format(cfg.limits.yaw_stop_deg))
    print("")
    print("Safety:")
    print("  enable_publish = {}".format(cfg.safety.enable_publish))
    print("  enable_scan_guard = {}".format(cfg.safety.enable_scan_guard))
    print("  enable_dynamic_obstacle_cost = {}".format(
        cfg.safety.enable_dynamic_obstacle_cost
    ))
    print("  front_stop_distance = {:.3f} m".format(
        cfg.safety.front_stop_distance
    ))
    print("")
    print("MPPI:")
    print("  use_existing_planner = {}".format(cfg.mppi.use_existing_planner))
    print("  use_anisotropic_sampling = {}".format(
        cfg.mppi.use_anisotropic_sampling
    ))
    print("  sdf_influence_distance = {:.3f}".format(
        cfg.mppi.sdf_influence_distance
    ))
    print("  scan_obstacle_mode = {}".format(cfg.mppi.scan_obstacle_mode))
    print("  scan_downsample_step = {}".format(cfg.mppi.scan_downsample_step))
    print("  dynamic_obstacle_max_count = {}".format(
        cfg.mppi.dynamic_obstacle_max_count
    ))
    print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to lab_runtime.yaml",
    )

    args = parser.parse_args()

    cfg = load_lab_runtime_config(args.config)
    print_config_summary(cfg)


if __name__ == "__main__":
    main()
