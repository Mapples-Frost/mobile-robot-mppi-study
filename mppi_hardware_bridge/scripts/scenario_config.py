from __future__ import print_function
from pathlib import Path

import argparse
import io
import os
import yaml

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "mppi_hardware_bridge" / "config" / "lab_runtime.yaml"


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

    if raw["limits"]["v_max"] > 0.20:
        raise ValueError(
            "limits.v_max is too large for the first hardware test. "
            "Please keep it <= 0.20 m/s."
        )

    if raw["limits"]["w_max"] > 0.80:
        raise ValueError(
            "limits.w_max is too large for the first hardware test. "
            "Please keep it <= 0.80 rad/s."
        )

    if raw["goal"]["tolerance"] <= 0.0:
        raise ValueError("goal.tolerance must be positive.")

    if raw["boundary"]["xmin"] >= raw["boundary"]["xmax"]:
        raise ValueError("boundary.xmin must be smaller than boundary.xmax.")

    if raw["boundary"]["ymin"] >= raw["boundary"]["ymax"]:
        raise ValueError("boundary.ymin must be smaller than boundary.ymax.")

    if raw["safety"]["front_stop_distance"] <= 0.0:
        raise ValueError("safety.front_stop_distance must be positive.")

    if raw["safety"]["front_slow_distance"] < raw["safety"]["front_stop_distance"]:
        raise ValueError(
            "safety.front_slow_distance should be >= front_stop_distance."
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
        raw = yaml.safe_load(f)

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