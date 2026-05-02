# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import math
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from control_adapter import control_to_twist_dict
from control_adapter import prepare_safe_command
from frame_transform import ExperimentFrameTransform
from frame_transform import normalize_angle
from local_obstacle_layer import limit_obstacles_evenly
from local_obstacle_layer import scan_to_experiment_obstacles
from local_obstacle_layer import scan_to_experiment_obstacles_geometric
from scan_guard import analyze_scan_front_sector


DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "config", "lab_runtime.yaml")
)


def print_ros_import_error(exc):
    print("")
    print("ROS import failed.")
    print("This script must be run in a ROS environment.")
    print("Original error: {}".format(exc))
    print("")


def print_config_error(exc):
    print("")
    print("Config load failed.")
    print("Please check lab_runtime.yaml and Python yaml support.")
    print("Original error: {}".format(exc))
    print("")


def print_planner_bridge_error(exc):
    print("")
    print("MPPI planner bridge initialization failed.")
    print("The adapter will not start in --planner-mode mppi.")
    print("Original error: {}".format(exc))
    print("")


def make_mppi_bounds_from_cfg(cfg):
    return {
        "x_min": float(cfg.boundary.xmin),
        "x_max": float(cfg.boundary.xmax),
        "y_min": float(cfg.boundary.ymin),
        "y_max": float(cfg.boundary.ymax),
    }


def quaternion_to_yaw(qx, qy, qz, qw):
    qx = float(qx)
    qy = float(qy)
    qz = float(qz)
    qw = float(qw)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return normalize_angle(math.atan2(siny_cosp, cosy_cosp))


def fake_planner_control(current_state_exp, goal_xy):
    """
    Temporary skeleton controller.

    Input:
      current_state_exp = (x_exp, y_exp, yaw_exp)
      goal_xy = (goal_x, goal_y)

    Output:
      (v, omega)

    This is not MPPI. It is only for adapter architecture review.
    """
    x, y, yaw = current_state_exp
    goal_x, goal_y = goal_xy

    dx = float(goal_x) - float(x)
    dy = float(goal_y) - float(y)
    distance = math.hypot(dx, dy)

    if distance < 1e-6:
        return (0.0, 0.0)

    desired_heading = math.atan2(dy, dx)
    heading_error = normalize_angle(desired_heading - float(yaw))

    if abs(heading_error) > 0.60:
        v = 0.0
        omega = 0.25 if heading_error > 0.0 else -0.25
    elif abs(heading_error) > 0.25:
        v = 0.02
        omega = max(-0.25, min(0.25, 0.8 * heading_error))
    else:
        v = min(0.05, 0.25 * distance)
        omega = max(-0.20, min(0.20, 0.6 * heading_error))

    return (v, omega)


def make_no_scan_result():
    return {
        "emergency_stop": True,
        "should_slow_down": False,
        "slow_scale": 0.0,
        "min_front_range": None,
        "min_side_range": None,
        "min_near_body_range": None,
        "valid_front_count": 0,
        "valid_side_count": 0,
        "valid_near_body_count": 0,
        "front_points": [],
        "side_points": [],
        "near_body_points": [],
        "raw_points_base": [],
        "side_avoid_mode": "none",
        "side_stop_mode": "none",
        "side_obstacle_side": "none",
        "side_soft_distance": None,
        "side_hard_distance": None,
        "side_release_distance": None,
        "omega_before_side_avoid": 0.0,
        "omega_after_side_avoid": 0.0,
        "side_avoid_applied": False,
        "side_hard_stop_applied": False,
        "front_stop_mode": "no_scan_emergency_stop",
        "front_stop_distance": None,
        "front_soft_block_distance": None,
        "hard_stop_distance": None,
        "front_slow_min_scale": None,
        "reason": "no_scan_emergency_stop",
    }


def make_twist_from_dict(twist_type, twist_dict):
    msg = twist_type()
    msg.linear.x = float(twist_dict["linear"]["x"])
    msg.linear.y = float(twist_dict["linear"]["y"])
    msg.linear.z = float(twist_dict["linear"]["z"])
    msg.angular.x = float(twist_dict["angular"]["x"])
    msg.angular.y = float(twist_dict["angular"]["y"])
    msg.angular.z = float(twist_dict["angular"]["z"])
    return msg


def clip_value(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def avoidance_side_to_omega_sign(side):
    if side == "left":
        return 1.0
    if side == "right":
        return -1.0
    return 0.0


def sign_of(value, epsilon=1e-6):
    value = float(value)
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def experiment_point_to_base(point, state):
    dx = float(point[0]) - float(state[0])
    dy = float(point[1]) - float(state[1])
    yaw = float(state[2])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        cos_yaw * dx + sin_yaw * dy,
        -sin_yaw * dx + cos_yaw * dy,
    )


class MppiRosAdapterSkeleton(object):
    def __init__(
        self,
        rospy,
        odom_type,
        scan_type,
        goal_type,
        twist_type,
        cfg,
        front_angle_offset_deg=0.0,
        rate_hz=5.0,
        enable_publish_override=None,
        planner_mode="fake",
    ):
        self.rospy = rospy
        self.twist_type = twist_type
        self.cfg = cfg
        if front_angle_offset_deg is None:
            front_angle_offset_deg = getattr(
                cfg.safety,
                "scan_angle_offset_deg",
                0.0,
            )
        self.front_angle_offset_deg = float(front_angle_offset_deg)
        self.base_frame_id = str(
            getattr(cfg.safety, "base_frame_id", "base_footprint")
        )
        self.rate_hz = max(1.0, float(rate_hz))
        self.planner_mode = str(planner_mode).strip().lower()
        self.mppi_bridge = None
        self.last_scan_frame_id = ""

        if self.planner_mode not in ("fake", "mppi"):
            raise ValueError("Unsupported planner_mode: {}".format(planner_mode))

        self.enable_publish = bool(cfg.safety.enable_publish)
        if enable_publish_override is not None:
            self.enable_publish = bool(enable_publish_override)

        self.frame_tf = ExperimentFrameTransform()
        self.current_state_exp = None
        self.last_raw_odom = None
        self.latest_odom_time = None
        self.latest_scan_result = None
        self.latest_scan_time = None
        self.latest_scan_msg_fields = None
        self.latest_local_obstacle_count = 0
        self.latest_dynamic_obstacle_count = 0
        self.latest_dynamic_obstacle_current_scan_count = 0
        self.latest_dynamic_obstacle_debug = {}
        self.avoidance_side = "none"
        self.avoidance_side_until = 0.0
        self.last_obstacle_centroid = None
        self.last_control_time = None
        self.last_smoothed_v = 0.0
        self.last_smoothed_omega = 0.0
        self.goal_reached_latched = False
        self.front_stop_recovery_active = False
        self.front_stop_recovery_direction = "none"
        self.obstacle_turn_active = False
        self.side_avoid_active = False
        self.side_avoid_until = 0.0
        self.side_avoid_side = "none"
        self.last_obstacle_turn_yaw_error_abs_deg = None
        self.latest_corridor_debug = {}
        self.avoidance_release_reason = "none"
        self.mppi_omega_trust_sign = 0
        self.mppi_omega_trust_count = 0
        self.front_soft_block_start_time = None
        self.previous_loop_start_time = None
        self.current_loop_start_time = None
        self.current_loop_dt_ms = None
        self.realtime_overrun_streak = 0
        self.realtime_degraded = False
        self.mppi_runtime_profile = "normal"
        self.mppi_degraded_reason = "normal"
        self.mppi_core_overrun_streak = 0
        self.mppi_core_recover_streak = 0
        self.avoidance_state = "CLEAR"
        self.avoidance_state_reason = "init"
        self.avoidance_state_since = time.time()
        self.clear_streak = 0
        self.progress_streak = 0
        self.stuck_streak = 0
        self.no_obstacle_streak = 0
        self.last_min_front_range = None
        self.last_goal_distance = None
        self.last_min_front_range_delta = None
        self.last_goal_distance_delta = None
        self.min_front_range_trend = "unknown"
        self.goal_distance_trend = "unknown"
        self.mppi_conflict_frames = 0
        self.hard_override_score_margin_streak = 0
        self.corridor_pending_switch_direction = "none"
        self.corridor_switch_streak = 0
        self.corridor_direction_since = 0.0
        self.corridor_direction_switched = False
        self.hard_stop_recovery_start_time = None
        self.hard_stop_recovery_frames = 0
        self.hard_stop_recovery_direction_since = 0.0
        self.hard_stop_recovery_no_improve_frames = 0
        self.hard_stop_recovery_last_min_front = None
        self.hard_stop_exit_to_creep = False
        self.line_surface_streak = 0
        self.latest_scan_tf_ok = True
        self.latest_scan_effective_offset_deg = self.front_angle_offset_deg
        self.tf_listener = None
        self.odom_timeout_sec = 1.0
        self.scan_timeout_sec = 1.0
        self.goal_xy = (float(cfg.goal.x), float(cfg.goal.y))
        self.debug_types = {}
        self.debug_pubs = {}
        self.debug_publish_last_time = {}
        self.cycle_status_count = 0
        self.last_cycle_status_signature = None

        try:
            from geometry_msgs.msg import Point
            from geometry_msgs.msg import Point32
            from sensor_msgs.msg import PointCloud
            from visualization_msgs.msg import Marker
            from visualization_msgs.msg import MarkerArray

            self.debug_types = {
                "Point32": Point32,
                "Point": Point,
                "PointCloud": PointCloud,
                "Marker": Marker,
                "MarkerArray": MarkerArray,
            }
            self.debug_pubs = {
                "raw_scan_points": rospy.Publisher(
                    "/mppi_debug/raw_scan_points",
                    PointCloud,
                    queue_size=1,
                ),
                "raw_scan_points_array": rospy.Publisher(
                    "/mppi_debug/raw_scan_points_array",
                    MarkerArray,
                    queue_size=1,
                ),
                "front_guard_points": rospy.Publisher(
                    "/mppi_debug/front_guard_points",
                    PointCloud,
                    queue_size=1,
                ),
                "geometric_obstacles_raw": rospy.Publisher(
                    "/mppi_debug/geometric_obstacles_raw",
                    MarkerArray,
                    queue_size=1,
                ),
                "planner_obstacles": rospy.Publisher(
                    "/mppi_debug/planner_obstacles",
                    MarkerArray,
                    queue_size=1,
                ),
                "goal_bearing": rospy.Publisher(
                    "/mppi_debug/goal_bearing",
                    Marker,
                    queue_size=1,
                ),
                "final_cmd": rospy.Publisher(
                    "/mppi_debug/final_cmd",
                    Marker,
                    queue_size=1,
                ),
            }
        except Exception as exc:
            rospy.logwarn(
                "[mppi_ros_adapter_skeleton] RViz debug publishers disabled: {}".format(
                    exc
                )
            )

        if bool(getattr(cfg.safety, "use_scan_tf", True)):
            try:
                import tf

                self.tf_listener = tf.TransformListener()
            except Exception as exc:
                rospy.logwarn(
                    "[mppi_ros_adapter_skeleton] tf listener unavailable; "
                    "non-base scan frames will fail safe. Original error: {}".format(exc)
                )

        if self.planner_mode == "mppi":
            try:
                from mppi_planner_bridge import MppiPlannerBridge

                self.mppi_bridge = MppiPlannerBridge(
                    cfg=cfg,
                    bounds=make_mppi_bounds_from_cfg(cfg),
                    obstacles=[],
                )
                rospy.logwarn(
                    "[mppi_ros_adapter_skeleton] planner_mode=mppi. "
                    "Real MPPI bridge is active for dry-run/control proposal."
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not initialize MppiPlannerBridge: {}".format(exc)
                )
        else:
            rospy.logwarn(
                "[mppi_ros_adapter_skeleton] planner_mode=fake. "
                "Using fake_planner_control placeholder."
            )

        # Lab safety reminder:
        # Before enabling publish, run:
        #   rostopic info /cmd_vel
        # Confirm no move_base or other controller is also publishing /cmd_vel.
        if self.enable_publish:
            self.cmd_pub = rospy.Publisher(
                cfg.ros.cmd_vel_topic,
                twist_type,
                queue_size=1,
            )
            rospy.logwarn(
                "[mppi_ros_adapter_skeleton] PUBLISH ENABLED on {}. "
                "Run 'rostopic info /cmd_vel' first and confirm no other controller "
                "is publishing.".format(cfg.ros.cmd_vel_topic)
            )
        else:
            self.cmd_pub = None
            rospy.logwarn(
                "[mppi_ros_adapter_skeleton] DRY-RUN mode. "
                "cfg.safety.enable_publish is False, so /cmd_vel will not be published."
            )

        self.odom_sub = rospy.Subscriber(
            cfg.ros.odom_topic,
            odom_type,
            self.odom_callback,
            queue_size=1,
        )
        self.scan_sub = rospy.Subscriber(
            cfg.ros.scan_topic,
            scan_type,
            self.scan_callback,
            queue_size=1,
        )
        self.goal_sub = rospy.Subscriber(
            cfg.ros.runtime_goal_topic,
            goal_type,
            self.runtime_goal_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "[mppi_ros_adapter_skeleton] odom={}, scan={}, goal={}, cmd_vel={}, "
            "planner_mode={}, base_frame={}, scan_angle_offset_deg={:.1f}. "
            "Before --enable-publish, run: rostopic info {}; confirm the base "
            "really consumes this topic and no smoother/move_base publisher overrides it.".format(
                cfg.ros.odom_topic,
                cfg.ros.scan_topic,
                cfg.ros.runtime_goal_topic,
                cfg.ros.cmd_vel_topic,
                self.planner_mode,
                self.base_frame_id,
                self.front_angle_offset_deg,
                cfg.ros.cmd_vel_topic,
            )
        )

    def odom_callback(self, msg):
        pose = msg.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        q = pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.last_raw_odom = (x, y, yaw)
        self.latest_odom_time = time.time()

        if not self.frame_tf.has_origin:
            self.frame_tf.set_origin(x, y, yaw)
            self.rospy.loginfo(
                "[mppi_ros_adapter_skeleton] Set experiment origin: "
                "x={:.3f}, y={:.3f}, yaw={:.3f}".format(x, y, yaw)
            )

        self.current_state_exp = self.frame_tf.odom_to_experiment(x, y, yaw)

    def make_debug_header(self, frame_id=None):
        header = type("HeaderLike", (), {})()
        header.stamp = self.rospy.Time.now()
        header.frame_id = frame_id or self.base_frame_id
        return header

    def debug_publish_allowed(self, topic_key, normal_period_sec, degraded_period_sec=None):
        debug_pubs = getattr(self, "debug_pubs", {})
        if topic_key not in debug_pubs:
            return False

        publisher = debug_pubs[topic_key]
        get_connections = getattr(publisher, "get_num_connections", None)
        if callable(get_connections):
            try:
                if int(get_connections()) <= 0:
                    return False
            except Exception:
                pass

        period = normal_period_sec
        if bool(getattr(self, "realtime_degraded", False)) and degraded_period_sec is not None:
            period = degraded_period_sec
        try:
            period = float(period)
        except (TypeError, ValueError):
            period = 0.0
        if period <= 0.0:
            return True

        last_times = getattr(self, "debug_publish_last_time", None)
        if last_times is None:
            last_times = {}
            self.debug_publish_last_time = last_times

        now = time.time()
        last_time = last_times.get(topic_key)
        if last_time is not None and now - float(last_time) < period:
            return False

        last_times[topic_key] = now
        return True

    def publish_point_cloud_debug(self, topic_key, points, color_mode=None):
        if topic_key not in self.debug_pubs or "PointCloud" not in self.debug_types:
            return
        if not self.debug_publish_allowed(topic_key, 0.50, 1.00):
            return

        cloud = self.debug_types["PointCloud"]()
        cloud.header.stamp = self.rospy.Time.now()
        cloud.header.frame_id = self.base_frame_id
        point_type = self.debug_types["Point32"]
        for point in points:
            msg_point = point_type()
            if isinstance(point, dict):
                msg_point.x = float(point.get("x", 0.0))
                msg_point.y = float(point.get("y", 0.0))
            else:
                msg_point.x = float(point[0]) if len(point) > 0 else 0.0
                msg_point.y = float(point[1]) if len(point) > 1 else 0.0
            msg_point.z = 0.02
            cloud.points.append(msg_point)

        self.debug_pubs[topic_key].publish(cloud)

    def publish_scan_points_array_debug(self, points):
        if (
            "raw_scan_points_array" not in self.debug_pubs
            or "MarkerArray" not in self.debug_types
        ):
            return
        if not self.debug_publish_allowed("raw_scan_points_array", 1.00, 2.00):
            return

        marker_type = self.debug_types["Marker"]
        array_type = self.debug_types["MarkerArray"]
        marker_array = array_type()

        delete_marker = marker_type()
        delete_marker.header.stamp = self.rospy.Time.now()
        delete_marker.header.frame_id = self.base_frame_id
        delete_marker.ns = "raw_scan_points_array"
        delete_marker.action = getattr(marker_type, "DELETEALL", 3)
        marker_array.markers.append(delete_marker)

        max_marker_count = 80 if bool(getattr(self, "realtime_degraded", False)) else 160
        sample_step = max(
            1,
            int(math.ceil(float(max(1, len(points))) / float(max_marker_count))),
        )
        marker_id = 1
        for index, point in enumerate(points):
            if index % sample_step != 0:
                continue
            marker = marker_type()
            marker.header.stamp = self.rospy.Time.now()
            marker.header.frame_id = self.base_frame_id
            marker.ns = "raw_scan_points_array"
            marker.id = marker_id
            marker_id += 1
            marker.type = getattr(marker_type, "SPHERE", 2)
            marker.action = getattr(marker_type, "ADD", 0)
            marker.pose.position.x = float(point["x"])
            marker.pose.position.y = float(point["y"])
            marker.pose.position.z = 0.02
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.035
            marker.scale.y = 0.035
            marker.scale.z = 0.035
            marker.color.r = 0.05
            marker.color.g = 0.25
            marker.color.b = 1.0
            marker.color.a = 0.75
            marker_array.markers.append(marker)

        self.debug_pubs["raw_scan_points_array"].publish(marker_array)

    def publish_obstacle_markers_debug(self, topic_key, obstacles, state, color):
        if topic_key not in self.debug_pubs or "MarkerArray" not in self.debug_types:
            return
        if not self.debug_publish_allowed(topic_key, 0.25, 0.75):
            return

        marker_type = self.debug_types["Marker"]
        array_type = self.debug_types["MarkerArray"]
        marker_array = array_type()

        delete_marker = marker_type()
        delete_marker.header.stamp = self.rospy.Time.now()
        delete_marker.header.frame_id = self.base_frame_id
        delete_marker.ns = topic_key
        delete_marker.action = getattr(marker_type, "DELETEALL", 3)
        marker_array.markers.append(delete_marker)

        for index, obstacle in enumerate(obstacles):
            marker = marker_type()
            marker.header.stamp = self.rospy.Time.now()
            marker.header.frame_id = self.base_frame_id
            marker.ns = topic_key
            marker.id = index + 1
            marker.type = getattr(marker_type, "SPHERE", 2)
            marker.action = getattr(marker_type, "ADD", 0)
            local_x, local_y = experiment_point_to_base((obstacle[0], obstacle[1]), state)
            radius = float(obstacle[2])
            marker.pose.position.x = local_x
            marker.pose.position.y = local_y
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2.0 * radius
            marker.scale.y = 2.0 * radius
            marker.scale.z = 0.08
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = float(color[3])
            marker_array.markers.append(marker)

        self.debug_pubs[topic_key].publish(marker_array)

    def publish_goal_bearing_debug(self, state, goal):
        if "goal_bearing" not in self.debug_pubs or "Marker" not in self.debug_types:
            return
        if not self.debug_publish_allowed("goal_bearing", 0.20, 0.75):
            return

        marker_type = self.debug_types["Marker"]
        point_type = self.debug_types["Point"]
        marker = marker_type()
        marker.header.stamp = self.rospy.Time.now()
        marker.header.frame_id = self.base_frame_id
        marker.ns = "goal_bearing"
        marker.id = 1
        marker.type = getattr(marker_type, "ARROW", 0)
        marker.action = getattr(marker_type, "ADD", 0)
        marker.pose.orientation.w = 1.0
        start = point_type()
        start.x = 0.0
        start.y = 0.0
        start.z = 0.08
        end = point_type()
        end_x, end_y = experiment_point_to_base(goal, state)
        length = math.hypot(end_x, end_y)
        if length > 1.2:
            end_x *= 1.2 / length
            end_y *= 1.2 / length
        end.x = end_x
        end.y = end_y
        end.z = 0.08
        marker.points.append(start)
        marker.points.append(end)
        marker.scale.x = 0.03
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.2
        marker.color.a = 0.9
        self.debug_pubs["goal_bearing"].publish(marker)

    def publish_final_cmd_debug(self, control, note):
        if "final_cmd" not in self.debug_pubs or "Marker" not in self.debug_types:
            return
        if not self.debug_publish_allowed("final_cmd", 0.20, 0.75):
            return

        marker_type = self.debug_types["Marker"]
        marker = marker_type()
        marker.header.stamp = self.rospy.Time.now()
        marker.header.frame_id = self.base_frame_id
        marker.ns = "final_cmd"
        marker.id = 1
        marker.type = getattr(marker_type, "TEXT_VIEW_FACING", 9)
        marker.action = getattr(marker_type, "ADD", 0)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.45
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.16
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = "v={:.2f} w={:.2f} {}".format(
            float(control[0]),
            float(control[1]),
            str(note),
        )
        self.debug_pubs["final_cmd"].publish(marker)

    def get_effective_scan_offset_deg(self, scan_frame_id):
        configured_offset = float(self.front_angle_offset_deg)
        if not scan_frame_id or scan_frame_id == self.base_frame_id:
            self.latest_scan_tf_ok = True
            return configured_offset

        if self.tf_listener is None:
            self.latest_scan_tf_ok = False
            self.rospy.logwarn(
                "[mppi_ros_adapter_skeleton] scan frame '{}' != base frame '{}' "
                "and tf listener is unavailable; fail-safe stop.".format(
                    scan_frame_id,
                    self.base_frame_id,
                )
            )
            return configured_offset

        try:
            _, rotation = self.tf_listener.lookupTransform(
                self.base_frame_id,
                scan_frame_id,
                self.rospy.Time(0),
            )
            yaw_scan_to_base = quaternion_to_yaw(
                rotation[0],
                rotation[1],
                rotation[2],
                rotation[3],
            )
            self.latest_scan_tf_ok = True
            return configured_offset - math.degrees(yaw_scan_to_base)
        except Exception as exc:
            self.latest_scan_tf_ok = False
            self.rospy.logwarn(
                "[mppi_ros_adapter_skeleton] TF lookup {} <- {} failed; "
                "fail-safe stop. Original error: {}".format(
                    self.base_frame_id,
                    scan_frame_id,
                    exc,
                )
            )
            return configured_offset

    def scan_callback(self, msg):
        scan_frame_id = getattr(getattr(msg, "header", None), "frame_id", "")
        self.last_scan_frame_id = scan_frame_id
        effective_offset_deg = self.get_effective_scan_offset_deg(scan_frame_id)
        self.latest_scan_effective_offset_deg = effective_offset_deg
        self.latest_scan_msg_fields = {
            "ranges": list(msg.ranges),
            "angle_min": msg.angle_min,
            "angle_increment": msg.angle_increment,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
            "frame_id": scan_frame_id,
            "effective_angle_offset_deg": effective_offset_deg,
        }
        if self.latest_scan_tf_ok:
            self.latest_scan_result = analyze_scan_front_sector(
                ranges=msg.ranges,
                angle_min=msg.angle_min,
                angle_increment=msg.angle_increment,
                range_min=msg.range_min,
                range_max=msg.range_max,
                front_stop_distance=self.cfg.safety.front_stop_distance,
                front_slow_distance=self.cfg.safety.front_slow_distance,
                front_angle_deg=self.cfg.safety.front_angle_deg,
                hard_stop_distance=getattr(
                    self.cfg.safety,
                    "hard_stop_distance",
                    self.cfg.safety.front_stop_distance,
                ),
                front_soft_block_distance=getattr(
                    self.cfg.safety,
                    "front_soft_block_distance",
                    self.cfg.safety.front_stop_distance,
                ),
                front_slow_min_scale=getattr(
                    self.cfg.safety,
                    "front_slow_min_scale",
                    0.45,
                ),
                front_angle_offset_deg=effective_offset_deg,
                side_stop_distance=getattr(self.cfg.safety, "side_stop_distance", 0.0),
                side_angle_deg=getattr(self.cfg.safety, "side_angle_deg", 120.0),
                near_body_stop_radius=getattr(
                    self.cfg.safety,
                    "near_body_stop_radius",
                    0.0,
                ),
                side_soft_distance=getattr(
                    self.cfg.safety,
                    "side_soft_distance",
                    getattr(self.cfg.safety, "side_stop_distance", 0.0),
                ),
                side_hard_distance=getattr(
                    self.cfg.safety,
                    "side_hard_distance",
                    getattr(self.cfg.safety, "side_stop_distance", 0.0),
                ),
                side_release_distance=getattr(
                    self.cfg.safety,
                    "side_release_distance",
                    getattr(self.cfg.safety, "side_stop_distance", 0.0),
                ),
                side_front_exclusion_angle_deg=getattr(
                    self.cfg.safety,
                    "side_front_exclusion_angle_deg",
                    25.0,
                ),
            )
        else:
            self.latest_scan_result = make_no_scan_result()
            self.latest_scan_result["reason"] = "scan_tf_failed"

        self.publish_point_cloud_debug(
            "raw_scan_points",
            self.latest_scan_result.get("raw_points_base", []),
        )
        self.publish_scan_points_array_debug(
            self.latest_scan_result.get("raw_points_base", [])
        )
        guard_points = list(self.latest_scan_result.get("front_points", []))
        guard_points.extend(self.latest_scan_result.get("side_points", []))
        guard_points.extend(self.latest_scan_result.get("near_body_points", []))
        self.publish_point_cloud_debug("front_guard_points", guard_points)
        self.latest_scan_time = time.time()

    def runtime_goal_callback(self, msg):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        frame_id = getattr(msg.header, "frame_id", "")
        self.rospy.logwarn(
            "[mppi_ros_adapter_skeleton] Runtime goal frame_id='{}'. "
            "This skeleton does not transform RViz goals yet. "
            "For the first hardware test, prefer the relative_to_start goal in "
            "lab_runtime.yaml to avoid frame mismatch.".format(frame_id)
        )
        self.goal_xy = (x, y)
        self.goal_reached_latched = False
        self.front_stop_recovery_active = False
        self.front_stop_recovery_direction = "none"
        self.rospy.loginfo(
            "[mppi_ros_adapter_skeleton] Updated runtime goal: "
            "x={:.3f}, y={:.3f}".format(x, y)
        )

    def is_odom_stale(self):
        if self.latest_odom_time is None:
            return True
        return (time.time() - self.latest_odom_time) > self.odom_timeout_sec

    def is_scan_stale(self):
        if self.latest_scan_time is None:
            return True
        return (time.time() - self.latest_scan_time) > self.scan_timeout_sec

    def clear_dynamic_obstacles(self):
        self.latest_local_obstacle_count = 0
        self.latest_dynamic_obstacle_count = 0
        self.latest_dynamic_obstacle_current_scan_count = 0
        self.latest_dynamic_obstacle_debug = {}
        self.last_obstacle_centroid = None

        if self.planner_mode != "mppi" or self.mppi_bridge is None:
            return

        if hasattr(self.mppi_bridge, "clear_obstacles"):
            self.mppi_bridge.clear_obstacles()
        else:
            self.mppi_bridge.set_obstacles([])

    def reset_control_filter(self):
        self.last_control_time = time.time()
        self.last_smoothed_v = 0.0
        self.last_smoothed_omega = 0.0

    def reset_omega_smoother(self, omega=0.0):
        self.last_control_time = time.time()
        self.last_smoothed_omega = float(omega)

    def ensure_runtime_state_defaults(self):
        now = time.time()
        defaults = {
            "avoidance_state": "CLEAR",
            "avoidance_state_reason": "init",
            "avoidance_state_since": now,
            "clear_streak": 0,
            "progress_streak": 0,
            "stuck_streak": 0,
            "no_obstacle_streak": 0,
            "last_min_front_range": None,
            "last_goal_distance": None,
            "last_min_front_range_delta": None,
            "last_goal_distance_delta": None,
            "min_front_range_trend": "unknown",
            "goal_distance_trend": "unknown",
            "mppi_conflict_frames": 0,
            "hard_override_score_margin_streak": 0,
            "corridor_pending_switch_direction": "none",
            "corridor_switch_streak": 0,
            "corridor_direction_since": 0.0,
            "corridor_direction_switched": False,
            "last_front_corridor_min_range": None,
            "front_corridor_worsening_streak": 0,
            "dynamic_corridor_candidate_streak": 0,
            "hard_stop_recovery_start_time": None,
            "hard_stop_recovery_frames": 0,
            "hard_stop_recovery_direction_since": 0.0,
            "hard_stop_recovery_no_improve_frames": 0,
            "hard_stop_recovery_last_min_front": None,
            "hard_stop_exit_to_creep": False,
            "line_surface_streak": 0,
            "mppi_runtime_profile": "normal",
            "mppi_degraded_reason": "normal",
            "mppi_core_overrun_streak": 0,
            "mppi_core_recover_streak": 0,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def clear_planner_obstacles_only(self):
        if self.planner_mode != "mppi" or self.mppi_bridge is None:
            return
        if hasattr(self.mppi_bridge, "clear_obstacles"):
            self.mppi_bridge.clear_obstacles()
        else:
            self.mppi_bridge.set_obstacles([])

    def set_avoidance_state(self, new_state, reason):
        self.ensure_runtime_state_defaults()
        now = time.time()
        if new_state != self.avoidance_state:
            self.avoidance_state = new_state
            self.avoidance_state_since = now
        self.avoidance_state_reason = reason

    def _trend_from_delta(self, delta, epsilon):
        if delta is None:
            return "unknown"
        if float(delta) > float(epsilon):
            return "increasing"
        if float(delta) < -float(epsilon):
            return "decreasing"
        return "stable"

    def update_progress_trends(self, scan_result, heading_debug):
        self.ensure_runtime_state_defaults()
        min_front_range = scan_result.get("min_front_range")
        goal_distance = None
        if heading_debug is not None:
            goal_distance = heading_debug.get("goal_distance")

        front_delta = None
        if min_front_range is not None and self.last_min_front_range is not None:
            front_delta = float(min_front_range) - float(self.last_min_front_range)
        goal_delta = None
        if goal_distance is not None and self.last_goal_distance is not None:
            goal_delta = float(goal_distance) - float(self.last_goal_distance)

        self.last_min_front_range_delta = front_delta
        self.last_goal_distance_delta = goal_delta
        self.min_front_range_trend = self._trend_from_delta(front_delta, 0.008)
        self.goal_distance_trend = self._trend_from_delta(goal_delta, 0.006)

        if min_front_range is not None:
            self.last_min_front_range = float(min_front_range)
        if goal_distance is not None:
            self.last_goal_distance = float(goal_distance)

        return {
            "min_front_range_trend": self.min_front_range_trend,
            "goal_distance_trend": self.goal_distance_trend,
            "min_front_range_delta": front_delta,
            "goal_distance_delta": goal_delta,
        }

    def update_avoidance_state(
        self,
        scan_result,
        heading_debug,
        planner_obstacles=0,
        dynamic_obstacles_current_scan=0,
    ):
        self.ensure_runtime_state_defaults()
        now = time.time()
        scan_reason = scan_result.get("reason", "")
        front_stop_mode = scan_result.get("front_stop_mode", scan_reason)
        min_front_range = scan_result.get("min_front_range")
        hard_stop_distance = float(getattr(self.cfg.safety, "hard_stop_distance", 0.18))
        front_soft_block_distance = float(
            getattr(self.cfg.safety, "front_soft_block_distance", 0.28)
        )
        front_slow_distance = float(getattr(self.cfg.safety, "front_slow_distance", 0.55))
        front_turn_release_distance = float(
            getattr(self.cfg.safety, "front_turn_release_distance", 0.95)
        )
        release_clear_frames = int(
            getattr(self.cfg.safety, "avoidance_release_clear_frames", 3)
        )
        progress_release_frames = int(
            getattr(self.cfg.safety, "avoidance_progress_release_frames", 3)
        )
        no_obstacle_release_frames = int(
            getattr(self.cfg.safety, "avoidance_no_obstacle_release_frames", 2)
        )
        release_margin = float(getattr(self.cfg.safety, "hard_stop_release_margin", 0.04))
        hard_stop_release_distance = hard_stop_distance + max(0.0, release_margin)
        preemptive_candidate_info = self.front_preemptive_candidate_info(
            scan_result,
            update_streaks=True,
        )

        min_front_value = None
        if min_front_range is not None:
            min_front_value = float(min_front_range)

        front_clear = (
            scan_reason == "front_clear"
            and min_front_value is not None
            and min_front_value > front_turn_release_distance
        )
        no_obstacles = (
            int(planner_obstacles) <= 0 and int(dynamic_obstacles_current_scan) <= 0
            and scan_reason == "front_clear"
        )
        release_candidate = bool(front_clear or no_obstacles)

        if release_candidate:
            self.clear_streak += 1
        else:
            self.clear_streak = 0

        if no_obstacles:
            self.no_obstacle_streak += 1
        else:
            self.no_obstacle_streak = 0

        front_not_worse = self.min_front_range_trend in ("increasing", "stable", "unknown")
        goal_progress = self.goal_distance_trend == "decreasing"
        yaw_ok = True
        if heading_debug is not None:
            yaw_ok = abs(float(heading_debug.get("yaw_error_deg", 0.0))) <= float(
                getattr(self.cfg.tracking, "yaw_align_deg", 115.0)
            )
        if goal_progress and front_not_worse and yaw_ok:
            self.progress_streak += 1
        else:
            self.progress_streak = 0

        if (
            self.goal_distance_trend in ("stable", "increasing")
            and self.min_front_range_trend == "decreasing"
        ):
            self.stuck_streak += 1
        else:
            self.stuck_streak = max(0, self.stuck_streak - 1)

        hard_stop_active = front_stop_mode in (
            "hard_stop",
            "front_obstacle_stop",
            "near_body_hard_stop",
        ) or scan_reason in ("hard_stop", "front_obstacle_stop", "near_body_hard_stop")
        hard_stop_release_candidate = (
            min_front_value is not None and min_front_value > hard_stop_release_distance
        )

        old_state = self.avoidance_state
        self.hard_stop_exit_to_creep = False
        new_state = "CLEAR"
        reason = "front_clear"
        release_block_reason = "none"

        if hard_stop_active:
            new_state = "HARD_STOP_RECOVERY"
            reason = "hard_stop"
            if self.hard_stop_recovery_start_time is None:
                self.hard_stop_recovery_start_time = now
        elif (
            old_state == "HARD_STOP_RECOVERY"
            and min_front_value is not None
            and min_front_value <= hard_stop_release_distance
        ):
            new_state = "HARD_STOP_RECOVERY"
            reason = "waiting_hard_stop_release_margin"
        elif min_front_value is not None and min_front_value <= front_soft_block_distance:
            new_state = "CREEP_ESCAPE"
            if old_state == "HARD_STOP_RECOVERY":
                self.hard_stop_exit_to_creep = True
                reason = "hard_stop_exit_to_creep"
            else:
                reason = "front_soft_block"
            self.hard_stop_recovery_start_time = None
        elif min_front_value is not None and min_front_value < front_slow_distance:
            new_state = "APPROACH_SLOW"
            reason = "front_obstacle_slow"
            self.hard_stop_recovery_start_time = None
        elif preemptive_candidate_info.get("front_corridor_worsening_streak", 0) >= max(
            1,
            int(getattr(self.cfg.safety, "preemptive_turn_candidate_frames", 2)),
        ):
            new_state = "APPROACH_SLOW"
            reason = "front_corridor_worsening"
            self.hard_stop_recovery_start_time = None
        elif preemptive_candidate_info.get("dynamic_corridor_candidate_streak", 0) >= max(
            1,
            int(getattr(self.cfg.safety, "preemptive_turn_candidate_frames", 2)),
        ):
            new_state = "APPROACH_SLOW"
            reason = "dynamic_corridor_sustained"
            self.hard_stop_recovery_start_time = None
        else:
            self.hard_stop_recovery_start_time = None
            if old_state in ("APPROACH_SLOW", "CREEP_ESCAPE", "HARD_STOP_RECOVERY"):
                if self.no_obstacle_streak >= max(1, no_obstacle_release_frames):
                    new_state = "GOAL_REACQUIRE"
                    reason = "no_obstacles_release"
                elif self.clear_streak >= max(1, release_clear_frames):
                    new_state = "GOAL_REACQUIRE"
                    reason = "front_turn_release_distance_clear"
                elif self.progress_streak >= max(1, progress_release_frames):
                    new_state = "GOAL_REACQUIRE"
                    reason = "progress_release"
                else:
                    new_state = old_state
                    reason = "release_conditions_pending"
                    if not release_candidate:
                        release_block_reason = "front_or_obstacles_not_clear"
                    elif self.progress_streak < max(1, progress_release_frames):
                        release_block_reason = "progress_streak_short"
            elif old_state == "GOAL_REACQUIRE":
                if self.clear_streak >= max(1, release_clear_frames):
                    new_state = "CLEAR"
                    reason = "goal_reacquire_complete"
                else:
                    new_state = "GOAL_REACQUIRE"
                    reason = "goal_reacquire_hold"
            else:
                new_state = "CLEAR"
                reason = "front_clear"

        self.set_avoidance_state(new_state, reason)
        if new_state in ("CLEAR", "GOAL_REACQUIRE"):
            self.avoidance_side = "none"
            self.avoidance_side_until = 0.0
            self.avoidance_release_reason = reason
            self.corridor_pending_switch_direction = "none"
            self.corridor_switch_streak = 0

        return self.current_avoidance_debug(
            release_candidate=release_candidate,
            release_block_reason=release_block_reason,
            hard_stop_release_candidate=hard_stop_release_candidate,
        )

    def current_avoidance_debug(
        self,
        release_candidate=None,
        release_block_reason=None,
        hard_stop_release_candidate=None,
    ):
        self.ensure_runtime_state_defaults()
        now = time.time()
        if release_candidate is None:
            release_candidate = self.clear_streak > 0 or self.no_obstacle_streak > 0
        if release_block_reason is None:
            release_block_reason = "none"
        if hard_stop_release_candidate is None:
            hard_stop_release_candidate = False
        return {
            "avoidance_state": self.avoidance_state,
            "avoidance_state_reason": self.avoidance_state_reason,
            "avoidance_state_age_sec": now - float(self.avoidance_state_since),
            "avoidance_release_candidate": bool(release_candidate),
            "avoidance_release_block_reason": release_block_reason,
            "clear_streak": int(self.clear_streak),
            "progress_streak": int(self.progress_streak),
            "stuck_streak": int(self.stuck_streak),
            "min_front_range_trend": self.min_front_range_trend,
            "goal_distance_trend": self.goal_distance_trend,
            "min_front_range_delta": self.last_min_front_range_delta,
            "goal_distance_delta": self.last_goal_distance_delta,
            "hard_stop_release_candidate": bool(hard_stop_release_candidate),
            "hard_stop_exit_to_creep": bool(self.hard_stop_exit_to_creep),
        }

    def _raw_scan_obstacles(self, state, max_radius, downsample_step, max_count):
        angle_offset_deg = self.latest_scan_msg_fields.get(
            "effective_angle_offset_deg",
            self.front_angle_offset_deg,
        )
        dynamic_obstacles = scan_to_experiment_obstacles(
            ranges=self.latest_scan_msg_fields["ranges"],
            angle_min=self.latest_scan_msg_fields["angle_min"],
            angle_increment=self.latest_scan_msg_fields["angle_increment"],
            range_min=self.latest_scan_msg_fields["range_min"],
            range_max=self.latest_scan_msg_fields["range_max"],
            current_state_exp=state,
            max_radius=max_radius,
            min_radius=0.10,
            angle_offset_rad=math.radians(angle_offset_deg),
            downsample_step=downsample_step,
            obstacle_radius=self.cfg.safety.local_obstacle_radius,
        )
        obstacle_count_before_limit = len(dynamic_obstacles)
        dynamic_obstacles = limit_obstacles_evenly(dynamic_obstacles, max_count)
        obstacle_debug = {
            "mode": "raw",
            "segment_count": 0,
            "edge_count": 0,
            "line_surface_count": 0,
            "circle_obstacle_count": 0,
            "edge_corner_count": 0,
            "irregular_count": 0,
            "obstacle_count_before_limit": obstacle_count_before_limit,
            "obstacle_count_after_limit": len(dynamic_obstacles),
            "geometric_obstacle_count_raw": obstacle_count_before_limit,
            "geometric_obstacle_count_used": len(dynamic_obstacles),
            "compressed_line_obstacles": 0,
            "line_compression_mode": "none",
            "line_fit_candidates": 0,
            "line_fit_accepted": 0,
            "line_fit_rejected_reason": "raw_mode",
            "line_surface_streak": 0,
            "representative_circle_count": 0,
            "obstacle_budget_mode": "raw_even_limit",
            "max_obstacle_count": max_count,
        }
        return dynamic_obstacles, obstacle_debug

    def _update_obstacle_centroid(self, dynamic_obstacles, state):
        if not dynamic_obstacles:
            self.last_obstacle_centroid = None
            return

        x_sum = 0.0
        y_sum = 0.0
        yaw = float(state[2])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        for obstacle in dynamic_obstacles:
            dx = float(obstacle[0]) - float(state[0])
            dy = float(obstacle[1]) - float(state[1])
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy
            x_sum += local_x
            y_sum += local_y
        count = float(len(dynamic_obstacles))
        self.last_obstacle_centroid = (x_sum / count, y_sum / count)

    def refresh_mppi_dynamic_obstacles(self, state, scan_unavailable):
        if self.planner_mode != "mppi" or self.mppi_bridge is None:
            return

        dynamic_enabled = bool(self.cfg.safety.enable_dynamic_obstacle_cost)
        if (
            scan_unavailable
            or self.latest_scan_msg_fields is None
            or not dynamic_enabled
        ):
            self.clear_dynamic_obstacles()
            return

        max_radius = getattr(
            self.cfg.mppi,
            "scan_obstacle_influence_distance",
            self.cfg.safety.front_slow_distance,
        )
        downsample_step = getattr(self.cfg.mppi, "scan_downsample_step", 3)
        mode = str(getattr(self.cfg.mppi, "scan_obstacle_mode", "raw")).strip().lower()
        if mode not in ("raw", "geometric"):
            self.rospy.logwarn(
                "[mppi_ros_adapter_skeleton] Unsupported scan_obstacle_mode='{}'. "
                "Falling back to raw.".format(mode)
            )
            mode = "raw"

        try:
            raw_max_obstacle_count = int(
                getattr(self.cfg.mppi, "dynamic_obstacle_max_count", 4)
            )
        except (TypeError, ValueError):
            raw_max_obstacle_count = 4

        try:
            geometric_max_obstacle_count = int(
                getattr(
                    self.cfg.mppi,
                    "geometric_max_obstacle_count",
                    raw_max_obstacle_count,
                )
            )
        except (TypeError, ValueError):
            geometric_max_obstacle_count = raw_max_obstacle_count

        max_obstacle_count = (
            geometric_max_obstacle_count if mode == "geometric" else raw_max_obstacle_count
        )
        max_obstacle_count = max(1, int(max_obstacle_count))

        try:
            line_max_obstacles = int(getattr(self.cfg.mppi, "geometric_line_max_obstacles", 3))
        except (TypeError, ValueError):
            line_max_obstacles = 3
        line_max_obstacles = max(2, line_max_obstacles)

        realtime_degraded = bool(getattr(self, "realtime_degraded", False))
        if realtime_degraded:
            downsample_step = max(2, int(downsample_step))
            max_obstacle_count = min(max_obstacle_count, 3)
            line_max_obstacles = min(line_max_obstacles, 2)

        try:
            if mode == "geometric":
                try:
                    dynamic_obstacles, obstacle_debug = scan_to_experiment_obstacles_geometric(
                        ranges=self.latest_scan_msg_fields["ranges"],
                        angle_min=self.latest_scan_msg_fields["angle_min"],
                        angle_increment=self.latest_scan_msg_fields["angle_increment"],
                        range_min=self.latest_scan_msg_fields["range_min"],
                        range_max=self.latest_scan_msg_fields["range_max"],
                        current_state_exp=state,
                        max_radius=max_radius,
                        min_radius=0.10,
                        angle_offset_rad=math.radians(
                            self.latest_scan_msg_fields.get(
                                "effective_angle_offset_deg",
                                self.front_angle_offset_deg,
                            )
                        ),
                        downsample_step=downsample_step,
                        smooth_window_size=getattr(
                            self.cfg.mppi,
                            "geometric_smooth_window_size",
                            3,
                        ),
                        obstacle_radius=self.cfg.safety.local_obstacle_radius,
                        max_neighbor_distance=getattr(
                            self.cfg.mppi,
                            "geometric_max_neighbor_distance",
                            0.15,
                        ),
                        tangent_jump_threshold_rad=getattr(
                            self.cfg.mppi,
                            "geometric_tangent_jump_threshold_rad",
                            0.55,
                        ),
                        curvature_threshold=getattr(
                            self.cfg.mppi,
                            "geometric_curvature_threshold",
                            2.5,
                        ),
                        curvature_delta_threshold=getattr(
                            self.cfg.mppi,
                            "geometric_curvature_delta_threshold",
                            2.0,
                        ),
                        line_error_threshold=getattr(
                            self.cfg.mppi,
                            "geometric_line_error_threshold",
                            0.035,
                        ),
                        line_curvature_threshold=getattr(
                            self.cfg.mppi,
                            "geometric_line_curvature_threshold",
                            0.35,
                        ),
                        min_line_length=getattr(
                            self.cfg.mppi,
                            "geometric_min_line_length",
                            0.15,
                        ),
                        circle_error_threshold=getattr(
                            self.cfg.mppi,
                            "geometric_circle_error_threshold",
                            0.035,
                        ),
                        circle_curvature_std_threshold=getattr(
                            self.cfg.mppi,
                            "geometric_circle_curvature_std_threshold",
                            0.60,
                        ),
                        min_circle_radius=getattr(
                            self.cfg.mppi,
                            "geometric_min_circle_radius",
                            0.04,
                        ),
                        max_circle_radius=getattr(
                            self.cfg.mppi,
                            "geometric_max_circle_radius",
                            0.60,
                        ),
                        min_circle_arc_angle=getattr(
                            self.cfg.mppi,
                            "geometric_min_circle_arc_angle",
                            0.35,
                        ),
                        line_sample_spacing=getattr(
                            self.cfg.mppi,
                            "geometric_line_spacing",
                            getattr(self.cfg.mppi, "geometric_line_sample_spacing", 0.22),
                        ),
                        line_max_obstacles=line_max_obstacles,
                        edge_corner_radius_scale=getattr(
                            self.cfg.mppi,
                            "geometric_edge_corner_radius_scale",
                            1.5,
                        ),
                        irregular_max_points=getattr(
                            self.cfg.mppi,
                            "geometric_irregular_max_points",
                            3,
                        ),
                        max_obstacle_count=max_obstacle_count,
                        min_segment_points=getattr(
                            self.cfg.mppi,
                            "geometric_min_segment_points",
                            4,
                        ),
                        obstacle_inflation=getattr(
                            self.cfg.mppi,
                            "geometric_obstacle_inflation",
                            0.04,
                        ),
                        return_debug=True,
                    )
                except Exception as exc:
                    self.rospy.logwarn(
                        "[mppi_ros_adapter_skeleton] geometric obstacle mode failed; "
                        "falling back to raw scan obstacles. Original error: {}".format(exc)
                    )
                    dynamic_obstacles, obstacle_debug = self._raw_scan_obstacles(
                        state=state,
                        max_radius=max_radius,
                        downsample_step=downsample_step,
                        max_count=raw_max_obstacle_count,
                    )
                    obstacle_debug["mode"] = "raw_fallback_from_geometric"
            else:
                dynamic_obstacles, obstacle_debug = self._raw_scan_obstacles(
                    state=state,
                    max_radius=max_radius,
                    downsample_step=downsample_step,
                    max_count=max_obstacle_count,
                )
        except Exception as exc:
            self.rospy.logwarn(
                "[mppi_ros_adapter_skeleton] scan obstacle generation failed; "
                "using empty dynamic obstacles. Original error: {}".format(exc)
            )
            dynamic_obstacles = []
            obstacle_debug = {
                "mode": "{}_failed_empty".format(mode),
                "segment_count": 0,
                "edge_count": 0,
                "line_surface_count": 0,
                "circle_obstacle_count": 0,
                "edge_corner_count": 0,
                "irregular_count": 0,
                "obstacle_count_before_limit": 0,
                "obstacle_count_after_limit": 0,
                "geometric_obstacle_count_raw": 0,
                "geometric_obstacle_count_used": 0,
                "compressed_line_obstacles": 0,
                "line_compression_mode": "none",
                "line_fit_candidates": 0,
                "line_fit_accepted": 0,
                "line_fit_rejected_reason": "generation_failed",
                "line_surface_streak": 0,
                "representative_circle_count": 0,
                "obstacle_budget_mode": "empty_failed",
                "max_obstacle_count": max_obstacle_count,
            }

        obstacle_debug["realtime_degraded"] = realtime_degraded
        obstacle_debug["effective_downsample_step"] = downsample_step
        obstacle_debug["effective_max_obstacle_count"] = max_obstacle_count
        self.ensure_runtime_state_defaults()
        if int(obstacle_debug.get("line_surface_count", 0)) > 0:
            self.line_surface_streak += 1
        else:
            self.line_surface_streak = max(0, self.line_surface_streak - 1)
        obstacle_debug["line_surface_streak"] = self.line_surface_streak
        if "representative_circle_count" not in obstacle_debug:
            obstacle_debug["representative_circle_count"] = obstacle_debug.get(
                "compressed_line_obstacles",
                0,
            )
        if "obstacle_budget_mode" not in obstacle_debug:
            obstacle_debug["obstacle_budget_mode"] = "bounded_even_limit"

        front_clear = (
            self.latest_scan_result is not None
            and self.latest_scan_result.get("reason") == "front_clear"
        )
        if front_clear and not dynamic_obstacles:
            self.clear_dynamic_obstacles()
            self.latest_dynamic_obstacle_debug = dict(obstacle_debug)
            self.publish_obstacle_markers_debug(
                "geometric_obstacles_raw",
                obstacle_debug.get("experiment_obstacles_before_limit", []),
                state,
                (1.0, 0.55, 0.0, 0.55),
            )
            self.publish_obstacle_markers_debug(
                "planner_obstacles",
                [],
                state,
                (1.0, 0.0, 0.0, 0.75),
            )
            return

        # The config name is historical. In this first pass, enabling it feeds
        # scan circles into the existing MPPI obstacles input; local_obstacle_cost.py
        # is intentionally not wired into trajectory_cost here.
        self.mppi_bridge.set_obstacles(dynamic_obstacles)
        self.latest_local_obstacle_count = len(dynamic_obstacles)
        self.latest_dynamic_obstacle_current_scan_count = len(dynamic_obstacles)
        self.latest_dynamic_obstacle_count = len(dynamic_obstacles)
        self.latest_dynamic_obstacle_debug = dict(obstacle_debug)
        self._update_obstacle_centroid(dynamic_obstacles, state)
        self.publish_obstacle_markers_debug(
            "geometric_obstacles_raw",
            obstacle_debug.get("experiment_obstacles_before_limit", dynamic_obstacles),
            state,
            (1.0, 0.55, 0.0, 0.55),
        )
        self.publish_obstacle_markers_debug(
            "planner_obstacles",
            dynamic_obstacles,
            state,
            (1.0, 0.0, 0.0, 0.75),
        )

    def scan_side_clearance(self):
        if self.latest_scan_msg_fields is None:
            return (None, None)

        fields = self.latest_scan_msg_fields
        front_angle_limit = math.radians(max(45.0, float(self.cfg.safety.front_angle_deg)))
        left_values = []
        right_values = []
        fallback_range = float(fields["range_max"])

        for index, raw_range in enumerate(fields["ranges"]):
            try:
                scan_range = float(raw_range)
            except (TypeError, ValueError):
                continue
            if math.isnan(scan_range) or math.isinf(scan_range):
                scan_range = fallback_range
            if scan_range < float(fields["range_min"]):
                continue
            scan_range = min(scan_range, fallback_range)
            angle = float(fields["angle_min"]) + float(index) * float(fields["angle_increment"])
            local_angle = normalize_angle(
                angle
                - math.radians(
                    fields.get(
                        "effective_angle_offset_deg",
                        self.front_angle_offset_deg,
                    )
                )
            )
            if abs(local_angle) > front_angle_limit:
                continue
            if local_angle >= 0.0:
                left_values.append(scan_range)
            else:
                right_values.append(scan_range)

        def mean_or_none(values):
            if not values:
                return None
            return sum(values) / float(len(values))

        return (mean_or_none(left_values), mean_or_none(right_values))

    def front_corridor_metrics(self, scan_result):
        front_turn_distance = float(getattr(self.cfg.safety, "front_turn_distance", 0.85))
        half_width = float(getattr(self.cfg.safety, "front_corridor_half_width", 0.38))
        corridor_points = []
        for point in scan_result.get("raw_points_base", []):
            x_value = float(point.get("x", 0.0))
            y_value = float(point.get("y", 0.0))
            if x_value > 0.15 and x_value < front_turn_distance and abs(y_value) < half_width:
                corridor_points.append(point)

        min_range = None
        for point in corridor_points:
            point_range = float(point.get("range", math.hypot(point.get("x", 0.0), point.get("y", 0.0))))
            if min_range is None or point_range < min_range:
                min_range = point_range

        return {
            "front_corridor_obstacle_count": len(corridor_points),
            "front_corridor_min_range": min_range,
            "front_corridor_points": corridor_points,
        }

    def front_preemptive_candidate_info(self, scan_result, update_streaks=False):
        self.ensure_runtime_state_defaults()
        min_front_range = scan_result.get("min_front_range")
        front_turn_distance = float(getattr(self.cfg.safety, "front_turn_distance", 0.85))
        front_slow_distance = float(getattr(self.cfg.safety, "front_slow_distance", 0.55))
        candidate_frames = int(
            getattr(self.cfg.safety, "preemptive_turn_candidate_frames", 2)
        )
        candidate_frames = max(1, candidate_frames)
        corridor = self.front_corridor_metrics(scan_result)
        corridor_min = corridor.get("front_corridor_min_range")
        corridor_count = int(corridor.get("front_corridor_obstacle_count", 0))
        previous_corridor_min = getattr(self, "last_front_corridor_min_range", None)

        min_front_value = None
        if min_front_range is not None:
            min_front_value = float(min_front_range)
        preemptive_turn_candidate = (
            min_front_value is not None and min_front_value < front_turn_distance
        )

        corridor_delta = None
        if corridor_min is not None and previous_corridor_min is not None:
            corridor_delta = float(corridor_min) - float(previous_corridor_min)
        corridor_worsening = (
            corridor_delta is not None and float(corridor_delta) < -0.008
        )
        dynamic_obstacles_current_scan = int(
            getattr(self, "latest_dynamic_obstacle_current_scan_count", 0)
        )
        dynamic_corridor_candidate = (
            dynamic_obstacles_current_scan > 0 and corridor_count > 0
        )

        if update_streaks:
            if corridor_worsening:
                self.front_corridor_worsening_streak += 1
            else:
                self.front_corridor_worsening_streak = 0
            if corridor_min is not None:
                self.last_front_corridor_min_range = float(corridor_min)
            elif corridor_count <= 0:
                self.last_front_corridor_min_range = None

            if dynamic_corridor_candidate:
                self.dynamic_corridor_candidate_streak += 1
            else:
                self.dynamic_corridor_candidate_streak = 0

        front_mode = scan_result.get(
            "front_stop_mode",
            scan_result.get("reason", ""),
        )
        scan_reason = scan_result.get("reason", "")
        front_slow_context = (
            front_mode in ("front_obstacle_slow", "front_soft_block")
            or scan_reason in ("front_obstacle_slow", "front_soft_block")
            or (
                min_front_value is not None
                and min_front_value < front_slow_distance
            )
        )
        corridor_worsening_allowed = (
            corridor_min is not None
            and self.front_corridor_worsening_streak >= candidate_frames
        )
        dynamic_corridor_allowed = (
            dynamic_corridor_candidate
            and self.dynamic_corridor_candidate_streak >= candidate_frames
        )

        candidate_reason = "none"
        if front_slow_context:
            candidate_reason = "front_slow_context"
        elif corridor_worsening_allowed:
            candidate_reason = "front_corridor_worsening"
        elif dynamic_corridor_allowed:
            candidate_reason = "dynamic_corridor_sustained"
        elif preemptive_turn_candidate:
            candidate_reason = "front_turn_distance_candidate"

        return {
            "preemptive_turn_candidate": bool(preemptive_turn_candidate),
            "obstacle_context_entry_allowed": bool(
                front_slow_context
                or corridor_worsening_allowed
                or dynamic_corridor_allowed
            ),
            "obstacle_context_gate_reason": candidate_reason,
            "front_slow_context": bool(front_slow_context),
            "front_corridor_worsening": bool(corridor_worsening),
            "front_corridor_delta": corridor_delta,
            "front_corridor_worsening_streak": int(
                self.front_corridor_worsening_streak
            ),
            "dynamic_corridor_candidate": bool(dynamic_corridor_candidate),
            "dynamic_corridor_candidate_streak": int(
                self.dynamic_corridor_candidate_streak
            ),
            "front_corridor_obstacle_count": corridor_count,
            "front_corridor_min_range": corridor_min,
        }

    def obstacle_turn_activation_info(self, scan_result, planner_obstacles=None):
        self.ensure_runtime_state_defaults()
        min_front_range = scan_result.get("min_front_range")
        front_turn_distance = float(getattr(self.cfg.safety, "front_turn_distance", 0.85))
        front_turn_release_distance = float(
            getattr(self.cfg.safety, "front_turn_release_distance", 0.95)
        )
        if planner_obstacles is None:
            planner_obstacles = self.latest_dynamic_obstacle_count

        corridor = self.front_corridor_metrics(scan_result)
        candidate_info = self.front_preemptive_candidate_info(
            scan_result,
            update_streaks=False,
        )
        scan_reason = scan_result.get("reason", "")
        front_stop_mode = scan_result.get("front_stop_mode", scan_reason)
        avoidance_state = str(getattr(self, "avoidance_state", "CLEAR"))
        active = False
        entry_reason = "none"
        release_reason = "none"
        release_state = avoidance_state in ("CLEAR", "GOAL_REACQUIRE")

        if release_state:
            release_reason = "release_state"
        elif (
            candidate_info.get("obstacle_context_entry_allowed", False)
            or front_stop_mode in ("front_obstacle_slow", "front_soft_block")
            or scan_reason in ("front_obstacle_slow", "front_soft_block")
        ):
            active = True
            entry_reason = candidate_info.get("obstacle_context_gate_reason", "none")
            if entry_reason == "none":
                entry_reason = front_stop_mode
        elif avoidance_state in ("APPROACH_SLOW", "CREEP_ESCAPE"):
            active = True
            entry_reason = "avoidance_state_hold"
        else:
            if (
                scan_reason == "front_clear"
                and min_front_range is not None
                and float(min_front_range) > front_turn_release_distance
            ):
                release_reason = "front_clear_release"
            elif (
                min_front_range is not None
                and float(min_front_range) > front_turn_release_distance
                and corridor["front_corridor_obstacle_count"] <= 0
            ):
                release_reason = "turn_corridor_clear"
            elif self.obstacle_turn_active:
                release_reason = "turn_corridor_clear"

        info = {
            "obstacle_turn_mode": bool(active),
            "obstacle_turn_entry_reason": entry_reason,
            "obstacle_turn_release_reason": release_reason,
            "front_corridor_obstacle_count": corridor["front_corridor_obstacle_count"],
            "front_corridor_min_range": corridor["front_corridor_min_range"],
        }
        info.update(candidate_info)
        return info

    def avoidance_is_active(self, scan_result):
        info = self.obstacle_turn_activation_info(scan_result)
        return bool(info.get("obstacle_turn_mode", False))

    def obstacle_turn_should_be_active(self, scan_result, planner_obstacles=None):
        info = self.obstacle_turn_activation_info(scan_result, planner_obstacles)
        return bool(info.get("obstacle_turn_mode", False))

    def side_risk_for_direction(self, direction, scan_result):
        side_soft_distance = float(getattr(self.cfg.safety, "side_soft_distance", 0.32))
        if side_soft_distance <= 0.0:
            return 0.0
        if direction == "left":
            side_min = scan_result.get("min_left_side_range")
        elif direction == "right":
            side_min = scan_result.get("min_right_side_range")
        else:
            side_min = None
        if side_min is None:
            return 0.0
        return clip_value(
            (side_soft_distance - float(side_min)) / max(side_soft_distance, 1e-6),
            0.0,
            1.0,
        )

    def compute_corridor_scores(self, scan_result, heading_debug=None):
        front_turn_distance = float(getattr(self.cfg.safety, "front_turn_distance", 0.85))
        half_width = float(getattr(self.cfg.safety, "front_corridor_half_width", 0.38))
        clearance_weight = float(getattr(self.cfg.safety, "corridor_clearance_weight", 1.0))
        goal_weight = float(getattr(self.cfg.safety, "corridor_goal_error_weight", 0.8))
        side_risk_weight = float(getattr(self.cfg.safety, "corridor_side_risk_weight", 1.5))
        max_yaw_error = float(getattr(self.cfg.safety, "obstacle_turn_max_yaw_error_deg", 22.0))

        corridor_points = self.front_corridor_metrics(scan_result)["front_corridor_points"]
        scores = {}
        debug = {}
        yaw_error = 0.0
        yaw_error_deg = 0.0
        if heading_debug:
            yaw_error = float(heading_debug.get("yaw_error", 0.0))
            yaw_error_deg = float(heading_debug.get("yaw_error_deg", 0.0))

        for direction in ("left", "right"):
            if direction == "left":
                side_points = [point for point in corridor_points if float(point.get("y", 0.0)) >= 0.0]
            else:
                side_points = [point for point in corridor_points if float(point.get("y", 0.0)) < 0.0]

            if side_points:
                min_range = min([float(point.get("range", front_turn_distance)) for point in side_points])
                min_range_score = clip_value(min_range / max(front_turn_distance, 1e-6), 0.0, 1.0)
                density_score = 1.0 - clip_value(float(len(side_points)) / 8.0, 0.0, 1.0)
                lateral_values = [abs(float(point.get("y", 0.0))) for point in side_points]
                width_score = clip_value(min(lateral_values) / max(half_width, 1e-6), 0.0, 1.0)
                clearance_score = clearance_weight * (
                    0.55 * min_range_score + 0.30 * density_score + 0.15 * width_score
                )
            else:
                clearance_score = clearance_weight

            turn_sign = avoidance_side_to_omega_sign(direction)
            if abs(yaw_error_deg) < 1e-6 or sign_of(yaw_error) == 0:
                goal_alignment_score = 0.0
            elif sign_of(yaw_error) == sign_of(turn_sign):
                goal_alignment_score = goal_weight * clip_value(
                    abs(yaw_error_deg) / max(max_yaw_error, 1e-6),
                    0.0,
                    1.0,
                )
            else:
                goal_alignment_score = -goal_weight * clip_value(
                    abs(yaw_error_deg) / max(max_yaw_error, 1e-6),
                    0.0,
                    1.0,
                )

            side_risk_score = side_risk_weight * self.side_risk_for_direction(
                direction,
                scan_result,
            )
            score = clearance_score + goal_alignment_score - side_risk_score
            scores[direction] = score
            debug["{}_corridor_score".format(direction)] = score

        return scores, debug

    def direction_pushes_toward_near_side(self, direction, scan_result):
        near_side = scan_result.get("side_obstacle_side", "none")
        if near_side == "left" and direction == "left":
            return True
        if near_side == "right" and direction == "right":
            return True
        return False

    def update_avoidance_side(self, scan_result, heading_debug=None, force_active=None):
        self.ensure_runtime_state_defaults()
        now = time.time()
        active = self.avoidance_is_active(scan_result) if force_active is None else bool(force_active)
        hold_sec = float(
            getattr(
                self.cfg.safety,
                "obstacle_turn_hold_sec",
                getattr(self.cfg.safety, "avoidance_side_hold_sec", 1.2),
            )
        )
        lock_min_sec = float(getattr(self.cfg.safety, "obstacle_turn_lock_min_sec", 0.25))
        lock_max_sec = float(getattr(self.cfg.safety, "obstacle_turn_lock_max_sec", 0.7))
        hold_sec = clip_value(hold_sec, lock_min_sec, max(lock_min_sec, lock_max_sec))
        margin = float(
            getattr(
                self.cfg.safety,
                "obstacle_turn_score_deadband",
                getattr(self.cfg.safety, "corridor_switch_margin", 0.15),
            )
        )
        scores, score_debug = self.compute_corridor_scores(scan_result, heading_debug)
        selected_side = "left" if scores.get("left", 0.0) >= scores.get("right", 0.0) else "right"
        old_side = self.avoidance_side
        switch_allowed = False
        switch_reason = "score_select"
        switch_blocked_by_lock = False
        switch_block_reason = "none"
        direction_switched = False
        score_margin_abs = abs(float(scores.get("left", 0.0)) - float(scores.get("right", 0.0)))
        required_switch_streak = int(
            getattr(self.cfg.safety, "corridor_switch_streak_frames", 3)
        )
        required_switch_streak = max(1, required_switch_streak)

        if not active:
            self.avoidance_side = "none"
            self.avoidance_side_until = 0.0
            self.avoidance_release_reason = "turn_corridor_clear"
            self.corridor_pending_switch_direction = "none"
            self.corridor_switch_streak = 0
            self.corridor_direction_switched = False
            score_debug.update({
                "corridor_selected_direction": "none",
                "corridor_switch_allowed": False,
                "corridor_switch_reason": self.avoidance_release_reason,
                "corridor_switch_margin": 0.0,
                "corridor_switch_streak": 0,
                "switch_blocked_by_lock": False,
                "switch_block_reason": "inactive",
                "selected_direction_age_sec": 0.0,
                "corridor_direction_switched": False,
            })
            self.latest_corridor_debug = score_debug
            return self.avoidance_side

        if old_side in ("left", "right"):
            old_score = scores.get(old_side, 0.0)
            new_score = scores.get(selected_side, 0.0)
            side_risk_bad = self.direction_pushes_toward_near_side(old_side, scan_result)
            if selected_side != old_side:
                score_delta = float(new_score) - float(old_score)
                strong_score_margin = score_delta >= margin
                sustained_side_risk = (
                    side_risk_bad
                    and score_delta >= margin * 0.50
                    and self.min_front_range_trend == "decreasing"
                )
                if strong_score_margin or sustained_side_risk:
                    if self.corridor_pending_switch_direction == selected_side:
                        self.corridor_switch_streak += 1
                    else:
                        self.corridor_pending_switch_direction = selected_side
                        self.corridor_switch_streak = 1
                else:
                    self.corridor_pending_switch_direction = "none"
                    self.corridor_switch_streak = 0

                lock_active = now < self.avoidance_side_until
                streak_ready = self.corridor_switch_streak >= required_switch_streak
                if (
                    streak_ready
                    and not lock_active
                    and (strong_score_margin or sustained_side_risk)
                ):
                    switch_allowed = True
                    switch_reason = (
                        "side_risk_override"
                        if sustained_side_risk
                        else "score_margin_streak"
                    )
                else:
                    selected_side = old_side
                    switch_allowed = False
                    if lock_active:
                        switch_reason = "hold_time"
                        switch_blocked_by_lock = True
                        switch_block_reason = "direction_lock"
                    elif not streak_ready:
                        switch_reason = "switch_streak_pending"
                        switch_block_reason = "corridor_switch_streak_short"
                    else:
                        switch_reason = "hysteresis_margin"
                        switch_block_reason = "score_margin"
            else:
                switch_reason = "same_direction"
                self.corridor_pending_switch_direction = "none"
                self.corridor_switch_streak = 0
                switch_allowed = True

        else:
            switch_allowed = True
            switch_reason = "initial_direction"
            self.corridor_pending_switch_direction = "none"
            self.corridor_switch_streak = 0

        if old_side in ("left", "right") and selected_side != old_side:
            direction_switched = True
            self.corridor_direction_since = now
        elif old_side not in ("left", "right"):
            self.corridor_direction_since = now

        self.avoidance_side = selected_side
        if switch_allowed or now >= self.avoidance_side_until:
            self.avoidance_side_until = now + hold_sec
        self.avoidance_release_reason = "locked"
        self.corridor_direction_switched = direction_switched
        selected_direction_age_sec = 0.0
        if self.corridor_direction_since > 0.0:
            selected_direction_age_sec = now - float(self.corridor_direction_since)
        score_debug.update({
            "corridor_selected_direction": selected_side,
            "corridor_switch_allowed": switch_allowed,
            "corridor_switch_reason": switch_reason,
            "corridor_switch_margin": score_margin_abs,
            "corridor_switch_streak": self.corridor_switch_streak,
            "switch_blocked_by_lock": switch_blocked_by_lock,
            "switch_block_reason": switch_block_reason,
            "selected_direction_age_sec": selected_direction_age_sec,
            "corridor_direction_switched": direction_switched,
        })
        self.latest_corridor_debug = score_debug
        return self.avoidance_side

    def apply_avoidance_hysteresis(self, control, scan_result):
        side = self.update_avoidance_side(scan_result)
        if side not in ("left", "right") or not self.avoidance_is_active(scan_result):
            return control, False

        v, omega = float(control[0]), float(control[1])
        desired_sign = avoidance_side_to_omega_sign(side)
        min_omega = float(getattr(self.cfg.safety, "front_turn_min_omega", 0.12))
        applied = False
        if desired_sign > 0.0 and omega < min_omega:
            omega = min_omega
            applied = True
        elif desired_sign < 0.0 and omega > -min_omega:
            omega = -min_omega
            applied = True
        return (v, omega), applied

    def avoidance_hold_remaining(self):
        if self.avoidance_side not in ("left", "right"):
            return 0.0
        return max(0.0, self.avoidance_side_until - time.time())

    def apply_avoidance_hysteresis(self, control, scan_result):
        side = self.update_avoidance_side(scan_result)
        if side not in ("left", "right") or not self.avoidance_is_active(scan_result):
            return control, False

        if (
            self.latest_dynamic_obstacle_count <= 1
            and scan_result.get("reason") not in ("front_obstacle_slow",)
        ):
            self.avoidance_side_until = min(
                self.avoidance_side_until,
                time.time() + 0.2,
            )
            return control, False

        v, omega = float(control[0]), float(control[1])
        desired_sign = avoidance_side_to_omega_sign(side)
        min_omega = float(getattr(self.cfg.safety, "front_turn_min_omega", 0.12))
        applied = False
        if desired_sign > 0.0 and omega < min_omega:
            omega = min_omega
            applied = True
        elif desired_sign < 0.0 and omega > -min_omega:
            omega = -min_omega
            applied = True
        return (v, omega), applied

    def obstacle_turn_omega_for_range(self, min_front_range):
        omega_min = float(getattr(self.cfg.safety, "front_turn_min_omega", 0.12))
        omega_max = float(getattr(self.cfg.safety, "front_turn_max_omega", 0.22))
        front_turn_distance = float(getattr(self.cfg.safety, "front_turn_distance", 0.85))
        front_slow_distance = float(getattr(self.cfg.safety, "front_slow_distance", 0.55))
        front_stop_distance = float(getattr(self.cfg.safety, "front_stop_distance", 0.30))

        if min_front_range is None:
            return omega_min
        distance = float(min_front_range)
        if distance >= front_turn_distance:
            return omega_min
        if distance <= front_stop_distance:
            return omega_max
        if distance >= front_slow_distance:
            ratio = (front_turn_distance - distance) / max(
                front_turn_distance - front_slow_distance,
                1e-6,
            )
            return omega_min + clip_value(ratio, 0.0, 1.0) * (0.18 - omega_min)
        ratio = (front_slow_distance - distance) / max(
            front_slow_distance - front_stop_distance,
            1e-6,
        )
        return 0.18 + clip_value(ratio, 0.0, 1.0) * (omega_max - 0.18)

    def omega_to_turn_direction(self, omega):
        omega_sign = sign_of(omega, 0.02)
        if omega_sign > 0:
            return "left"
        if omega_sign < 0:
            return "right"
        return "none"

    def update_mppi_omega_trust(self, omega):
        omega_sign = sign_of(omega, 0.02)
        if omega_sign == 0:
            return self.mppi_omega_trust_count, self.mppi_omega_trust_sign

        if omega_sign == self.mppi_omega_trust_sign:
            self.mppi_omega_trust_count += 1
        else:
            self.mppi_omega_trust_sign = omega_sign
            self.mppi_omega_trust_count = 1
        return self.mppi_omega_trust_count, self.mppi_omega_trust_sign

    def obstacle_turn_risk_score(self, scan_result, activation_info):
        min_front_range = scan_result.get("min_front_range")
        front_turn_distance = float(getattr(self.cfg.safety, "front_turn_distance", 0.85))
        hard_stop_distance = float(getattr(self.cfg.safety, "hard_stop_distance", 0.18))
        score = 0.0

        if min_front_range is not None:
            denom = max(front_turn_distance - hard_stop_distance, 1e-6)
            score = (front_turn_distance - float(min_front_range)) / denom
            score = clip_value(score, 0.0, 1.0)

        reason = scan_result.get("reason", "")
        front_stop_mode = scan_result.get("front_stop_mode", reason)
        if front_stop_mode == "front_soft_block" or reason == "front_soft_block":
            score = max(score, 0.70)
        elif front_stop_mode == "front_obstacle_slow" or reason == "front_obstacle_slow":
            score = max(score, 0.45)
        elif front_stop_mode in ("hard_stop", "near_body_hard_stop"):
            score = 1.0

        corridor_count = int(activation_info.get("front_corridor_obstacle_count", 0))
        if corridor_count > 0:
            score = max(score, clip_value(float(corridor_count) / 6.0, 0.0, 0.80))

        return clip_value(score, 0.0, 1.0)

    def apply_preemptive_obstacle_turn(
        self,
        control,
        scan_result,
        planner_obstacles,
        heading_debug=None,
        activation_info=None,
    ):
        self.ensure_runtime_state_defaults()
        min_front_range = scan_result.get("min_front_range")
        front_turn_distance = float(getattr(self.cfg.safety, "front_turn_distance", 0.85))
        front_turn_release_distance = float(
            getattr(self.cfg.safety, "front_turn_release_distance", 0.95)
        )
        keep_v_scale = float(getattr(self.cfg.safety, "front_turn_keep_v_scale", 0.75))
        keep_v_scale = clip_value(keep_v_scale, 0.0, 1.0)

        if activation_info is None:
            activation_info = self.obstacle_turn_activation_info(
                scan_result,
                planner_obstacles,
            )
        active = bool(activation_info.get("obstacle_turn_mode", False))
        release_state = self.avoidance_state in ("CLEAR", "GOAL_REACQUIRE")
        if release_state:
            active = False
            activation_info["obstacle_turn_mode"] = False
            activation_info["obstacle_turn_release_reason"] = "release_state"
        self.obstacle_turn_active = bool(active)

        debug = {
            "obstacle_turn_mode": bool(active),
            "front_turn_distance": front_turn_distance,
            "front_turn_release_distance": front_turn_release_distance,
            "front_corridor_obstacle_count": activation_info.get(
                "front_corridor_obstacle_count",
                0,
            ),
            "front_corridor_min_range": activation_info.get(
                "front_corridor_min_range",
                None,
            ),
            "preemptive_turn_candidate": activation_info.get(
                "preemptive_turn_candidate",
                False,
            ),
            "obstacle_context_entry_allowed": activation_info.get(
                "obstacle_context_entry_allowed",
                False,
            ),
            "obstacle_context_gate_reason": activation_info.get(
                "obstacle_context_gate_reason",
                "none",
            ),
            "front_corridor_worsening_streak": activation_info.get(
                "front_corridor_worsening_streak",
                0,
            ),
            "dynamic_corridor_candidate_streak": activation_info.get(
                "dynamic_corridor_candidate_streak",
                0,
            ),
            "obstacle_turn_entry_reason": activation_info.get(
                "obstacle_turn_entry_reason",
                "none",
            ),
            "obstacle_turn_release_reason": activation_info.get(
                "obstacle_turn_release_reason",
                "none",
            ),
            "obstacle_turn_yaw_damped": False,
            "obstacle_turn_yaw_released": False,
            "obstacle_turn_direction": "none",
            "obstacle_turn_omega": 0.0,
            "omega_before_obstacle_turn": float(control[1]),
            "omega_after_obstacle_turn": float(control[1]),
            "omega_source": "mppi",
            "arbitration_mode": "mppi",
            "hard_override_allowed": False,
            "hard_override_gate_reason": "inactive",
            "mppi_direction_conflict": False,
            "mppi_conflict_frames": self.mppi_conflict_frames,
            "min_front_range_trend": self.min_front_range_trend,
            "goal_distance_trend": self.goal_distance_trend,
            "mppi_omega_preserved": True,
            "mppi_omega_overridden": False,
            "omega_override_reason": "none",
            "proposed_omega": float(control[1]),
            "mppi_turn_direction": self.omega_to_turn_direction(control[1]),
            "override_risk_score": 0.0,
            "mppi_omega_trust_count": self.mppi_omega_trust_count,
            "turn_v_scale": 1.0,
            "avoidance_direction_locked": self.avoidance_side in ("left", "right"),
            "avoidance_release_reason": self.avoidance_release_reason,
            "left_corridor_score": 0.0,
            "right_corridor_score": 0.0,
            "corridor_selected_direction": "none",
            "corridor_switch_allowed": False,
            "corridor_switch_reason": "none",
            "corridor_switch_margin": 0.0,
            "corridor_switch_streak": 0,
            "switch_blocked_by_lock": False,
            "switch_block_reason": "none",
            "selected_direction_age_sec": 0.0,
        }
        debug.update(self.current_avoidance_debug())
        if not active:
            release_reason = activation_info.get("obstacle_turn_release_reason", "none")
            if release_reason != "none":
                self.avoidance_release_reason = release_reason
                self.avoidance_side = "none"
                self.avoidance_side_until = 0.0
                self.obstacle_turn_active = False
                self.mppi_conflict_frames = 0
                self.hard_override_score_margin_streak = 0
                self.corridor_pending_switch_direction = "none"
                self.corridor_switch_streak = 0
            debug["avoidance_release_reason"] = self.avoidance_release_reason
            debug["avoidance_direction_locked"] = self.avoidance_side in ("left", "right")
            debug["mppi_conflict_frames"] = self.mppi_conflict_frames
            debug["hard_override_score_margin_streak"] = self.hard_override_score_margin_streak
            debug["corridor_selected_direction"] = "none"
            debug["corridor_switch_reason"] = release_reason
            return control, debug

        side = self.update_avoidance_side(
            scan_result,
            heading_debug=heading_debug,
            force_active=True,
        )
        debug.update(self.latest_corridor_debug)
        if side not in ("left", "right"):
            side = "left"
            self.avoidance_side = side
            self.avoidance_side_until = time.time() + float(
                getattr(self.cfg.safety, "obstacle_turn_hold_sec", 1.2)
            )

        desired_sign = avoidance_side_to_omega_sign(side)
        omega_turn = self.obstacle_turn_omega_for_range(min_front_range)
        override_risk_score = self.obstacle_turn_risk_score(scan_result, activation_info)
        override_min_risk = float(
            getattr(self.cfg.safety, "obstacle_turn_override_min_risk", 0.65)
        )
        override_deadband = float(
            getattr(self.cfg.safety, "obstacle_turn_score_deadband", 0.20)
        )
        bias_omega = float(getattr(self.cfg.safety, "obstacle_turn_bias_omega", 0.06))
        trust_frames = int(getattr(self.cfg.safety, "mppi_omega_trust_frames", 3))
        yaw_error = 0.0
        yaw_error_abs_deg = 0.0
        if heading_debug:
            yaw_error = float(heading_debug.get("yaw_error", 0.0))
            yaw_error_abs_deg = abs(float(heading_debug.get("yaw_error_deg", 0.0)))
        damp_yaw_error_deg = float(
            getattr(self.cfg.safety, "obstacle_turn_damp_yaw_error_deg", 14.0)
        )
        max_yaw_error_deg = float(
            getattr(self.cfg.safety, "obstacle_turn_max_yaw_error_deg", 22.0)
        )
        turn_worsens_yaw = (
            sign_of(yaw_error) != 0
            and sign_of(desired_sign) != 0
            and sign_of(yaw_error) != sign_of(desired_sign)
        )
        front_slow_distance = float(getattr(self.cfg.safety, "front_slow_distance", 0.55))
        front_dangerous = (
            scan_result.get("reason") in (
                "front_obstacle_slow",
                "front_soft_block",
                "hard_stop",
                "front_obstacle_stop",
            )
            or (
                min_front_range is not None
                and float(min_front_range) < front_slow_distance
            )
        )
        if turn_worsens_yaw and yaw_error_abs_deg > max_yaw_error_deg and not front_dangerous:
            self.avoidance_release_reason = "yaw_error_release"
            self.avoidance_side = "none"
            self.avoidance_side_until = 0.0
            self.obstacle_turn_active = False
            self.last_obstacle_turn_yaw_error_abs_deg = yaw_error_abs_deg
            debug.update(
                {
                    "obstacle_turn_mode": False,
                    "obstacle_turn_direction": "none",
                    "obstacle_turn_release_reason": "yaw_error_release",
                    "obstacle_turn_yaw_released": True,
                    "avoidance_direction_locked": False,
                    "avoidance_release_reason": self.avoidance_release_reason,
                }
            )
            return control, debug

        if turn_worsens_yaw and yaw_error_abs_deg > damp_yaw_error_deg:
            if max_yaw_error_deg <= damp_yaw_error_deg:
                damp_scale = 0.0
            else:
                damp_scale = (max_yaw_error_deg - yaw_error_abs_deg) / max(
                    max_yaw_error_deg - damp_yaw_error_deg,
                    1e-6,
                )
            omega_turn *= clip_value(damp_scale, 0.0, 1.0)
            debug["obstacle_turn_yaw_damped"] = True

        if self.avoidance_state == "CREEP_ESCAPE":
            creep_limit = float(getattr(self.cfg.safety, "planner_creep_max_omega", 0.14))
            omega_turn = min(abs(omega_turn), max(0.0, creep_limit))

        desired_omega = desired_sign * omega_turn
        v = float(control[0])
        omega_before = float(control[1])
        mppi_turn_direction = self.omega_to_turn_direction(omega_before)
        trust_count, trust_sign = self.update_mppi_omega_trust(omega_before)
        omega_source = "mppi"
        arbitration_mode = "mppi"
        mppi_omega_preserved = True
        mppi_omega_overridden = False
        omega_override_reason = "none"
        hard_override_allowed = False
        hard_override_gate_reason = "no_conflict"
        mppi_direction_conflict = (
            sign_of(omega_before) != 0
            and sign_of(desired_omega) != 0
            and sign_of(omega_before) != sign_of(desired_omega)
        )
        if mppi_direction_conflict:
            self.mppi_conflict_frames += 1
        else:
            self.mppi_conflict_frames = 0

        corridor_margin = float(
            self.latest_corridor_debug.get("corridor_switch_margin", 0.0)
        )
        required_margin_streak = int(
            getattr(self.cfg.safety, "hard_override_margin_streak_frames", 3)
        )
        required_conflict_streak = int(
            getattr(self.cfg.safety, "hard_override_conflict_streak_frames", 2)
        )
        required_margin_streak = max(1, required_margin_streak)
        required_conflict_streak = max(1, required_conflict_streak)
        if corridor_margin >= max(override_deadband, 1e-6):
            self.hard_override_score_margin_streak += 1
        else:
            self.hard_override_score_margin_streak = 0

        front_stop_distance = float(getattr(self.cfg.safety, "front_stop_distance", 0.30))
        front_soft_block_distance = float(
            getattr(self.cfg.safety, "front_soft_block_distance", front_stop_distance)
        )
        hard_gate_release_state = self.avoidance_state in ("GOAL_REACQUIRE", "CLEAR")
        front_close = (
            min_front_range is not None
            and float(min_front_range) <= front_stop_distance + 0.04
        )
        front_block_close = (
            min_front_range is not None
            and float(min_front_range) <= front_soft_block_distance
        )
        front_worsening = self.min_front_range_trend == "decreasing"
        goal_not_progressing = self.goal_distance_trend in ("stable", "increasing", "unknown")
        if not mppi_direction_conflict:
            hard_override_gate_reason = "no_direction_conflict"
        elif hard_gate_release_state:
            hard_override_gate_reason = "release_state"
        elif override_risk_score < override_min_risk:
            hard_override_gate_reason = "risk_below_override_threshold"
        elif not (front_worsening or front_close or front_block_close):
            hard_override_gate_reason = "front_range_not_worsening_or_close"
        elif self.hard_override_score_margin_streak < required_margin_streak:
            hard_override_gate_reason = "score_margin_streak_short"
        elif self.mppi_conflict_frames < required_conflict_streak:
            hard_override_gate_reason = "mppi_conflict_streak_short"
        elif not (front_worsening or goal_not_progressing):
            hard_override_gate_reason = "mppi_direction_not_hurting_progress"
        else:
            hard_override_allowed = True
            hard_override_gate_reason = "sustained_high_risk_conflict"

        if abs(desired_omega) <= 1e-6 or desired_sign == 0.0:
            omega = omega_before
        elif sign_of(omega_before) == sign_of(desired_omega):
            biased_abs = abs(omega_before) + max(0.0, bias_omega)
            upper_abs = max(abs(omega_before), abs(desired_omega))
            omega = desired_sign * clip_value(biased_abs, abs(omega_before), upper_abs)
            if abs(omega - omega_before) > 1e-6:
                omega_source = "mppi_biased"
                arbitration_mode = "bias"
                omega_override_reason = "same_direction_bias"
        elif sign_of(omega_before) == 0:
            omega = desired_sign * min(max(bias_omega, 0.5 * omega_turn), omega_turn)
            omega_source = "mppi_biased"
            arbitration_mode = "bias"
            omega_override_reason = "zero_mppi_omega_bias"
        else:
            trust_blocks_override = (
                trust_frames > 0
                and trust_count >= trust_frames
                and trust_sign == sign_of(omega_before)
                and override_risk_score < override_min_risk + override_deadband
            )
            if hard_override_allowed and not trust_blocks_override:
                omega = desired_omega
                omega_source = "hard_override"
                arbitration_mode = "hard_override"
                mppi_omega_preserved = False
                mppi_omega_overridden = True
                omega_override_reason = hard_override_gate_reason
            else:
                if self.avoidance_state == "CREEP_ESCAPE":
                    sign_preserve_limit = float(
                        getattr(self.cfg.safety, "planner_creep_max_omega", 0.14)
                    )
                else:
                    sign_preserve_limit = float(
                        getattr(self.cfg.safety, "front_turn_max_omega", 0.18)
                    )
                sign_preserve_limit = min(
                    abs(sign_preserve_limit),
                    float(self.cfg.limits.w_max),
                )
                omega = sign_of(omega_before) * min(abs(omega_before), sign_preserve_limit)
                if abs(omega - omega_before) > 1e-6:
                    arbitration_mode = "clamp"
                    omega_source = "mppi_clamped"
                omega_override_reason = (
                    "mppi_trust_preserved"
                    if trust_blocks_override
                    else hard_override_gate_reason
                )
                if override_risk_score >= override_min_risk:
                    v = min(v, max(float(getattr(self.cfg.safety, "soft_avoid_min_v", 0.055)), v * 0.85))

        omega = clip_value(
            omega,
            -float(self.cfg.limits.w_max),
            float(self.cfg.limits.w_max),
        )

        if min_front_range is not None and float(min_front_range) >= front_slow_distance:
            v = max(v, float(self.cfg.limits.v_max) * keep_v_scale)
            turn_v_scale = keep_v_scale
        else:
            turn_v_scale = 1.0

        debug.update(
            {
                "obstacle_turn_direction": side,
                "obstacle_turn_omega": omega_turn,
                "omega_after_obstacle_turn": omega,
                "omega_source": omega_source,
                "arbitration_mode": arbitration_mode,
                "hard_override_allowed": hard_override_allowed,
                "hard_override_gate_reason": hard_override_gate_reason,
                "mppi_direction_conflict": mppi_direction_conflict,
                "mppi_conflict_frames": self.mppi_conflict_frames,
                "min_front_range_trend": self.min_front_range_trend,
                "goal_distance_trend": self.goal_distance_trend,
                "mppi_omega_preserved": mppi_omega_preserved,
                "mppi_omega_overridden": mppi_omega_overridden,
                "omega_override_reason": omega_override_reason,
                "proposed_omega": omega_before,
                "mppi_turn_direction": mppi_turn_direction,
                "override_risk_score": override_risk_score,
                "mppi_omega_trust_count": trust_count,
                "hard_override_score_margin_streak": self.hard_override_score_margin_streak,
                "turn_v_scale": turn_v_scale,
                "avoidance_direction_locked": self.avoidance_side in ("left", "right"),
                "avoidance_release_reason": self.avoidance_release_reason,
            }
        )
        self.last_obstacle_turn_yaw_error_abs_deg = yaw_error_abs_deg
        return (v, omega), debug

    def compute_heading_debug(self, state, goal):
        dx = float(goal[0]) - float(state[0])
        dy = float(goal[1]) - float(state[1])
        goal_distance = math.hypot(dx, dy)
        if goal_distance <= 1e-9:
            goal_bearing = float(state[2])
            yaw_error = 0.0
        else:
            goal_bearing = math.atan2(dy, dx)
            yaw_error = normalize_angle(goal_bearing - float(state[2]))
        return {
            "goal_distance": goal_distance,
            "goal_bearing": goal_bearing,
            "yaw_error": yaw_error,
            "yaw_error_deg": math.degrees(yaw_error),
        }

    def _apply_goal_tracking_override(self, state, goal, control, config, obstacle_context):
        heading_debug = self.compute_heading_debug(state, goal)
        yaw_error = float(heading_debug["yaw_error"])
        abs_yaw_error_deg = abs(float(heading_debug["yaw_error_deg"]))
        tracking = getattr(config, "tracking", None)
        if tracking is None or not bool(getattr(tracking, "enable_goal_tracking_override", True)):
            debug = dict(heading_debug)
            debug.update({
                "tracking_mode": "disabled",
                "tracking_v_scale": 1.0,
                "omega_before_tracking": float(control[1]),
                "omega_after_tracking": float(control[1]),
                "tracking_omega_before": float(control[1]),
                "tracking_omega_after": float(control[1]),
                "goal_tracking_intervened": False,
                "goal_tracking_reason": "disabled",
                "smoothing_reset_reason": "none",
                "obstacle_yaw_deadband_escape_applied": False,
                "goal_reacquire_active": False,
                "anti_goal_drive_blocked": False,
                "heading_gate_reason": "disabled",
                "goal_tracking_suppression_release_reason": "disabled",
            })
            return control, debug

        yaw_deadband = float(getattr(tracking, "yaw_deadband_deg", 6.0))
        yaw_slow = float(getattr(tracking, "yaw_slow_deg", 18.0))
        yaw_align = float(getattr(tracking, "yaw_align_deg", 32.0))
        k_yaw = float(getattr(tracking, "k_yaw", 0.75))
        omega_track_max = float(getattr(tracking, "omega_track_max", 0.45))
        omega_align_max = float(getattr(tracking, "omega_align_max", 0.55))
        omega_deadband_max = float(getattr(tracking, "omega_deadband_max", 0.12))
        min_v_scale = float(getattr(tracking, "min_v_scale", 0.25))
        suppress_wrong_sign = bool(getattr(tracking, "suppress_wrong_sign_omega", True))
        reset_on_flip = bool(getattr(tracking, "reset_smoother_on_omega_sign_flip", True))

        omega_before = float(control[1])
        v = float(control[0])
        omega = omega_before
        tracking_mode = "pass_through"
        v_scale = 1.0
        smoothing_reset_reason = "none"
        goal_tracking_intervened = False
        goal_tracking_reason = "none"
        tracking_omega_limit = omega_align_max
        obstacle_yaw_deadband_escape_applied = False
        scan_result = obstacle_context.get("scan_result", {}) if obstacle_context else {}
        avoidance_state = "CLEAR"
        planner_obstacles = 0
        dynamic_obstacles_current_scan = 0
        obstacle_mode = str(getattr(config.mppi, "scan_obstacle_mode", "raw")).strip().lower()
        proposed_omega = None
        if obstacle_context:
            planner_obstacles = int(
                obstacle_context.get(
                    "planner_obstacles",
                    obstacle_context.get("dynamic_obstacles", 0),
                )
            )
            dynamic_obstacles_current_scan = int(
                obstacle_context.get(
                    "dynamic_obstacles_current_scan",
                    obstacle_context.get("dynamic_obstacles", 0),
                )
            )
            obstacle_mode = str(
                obstacle_context.get("obstacle_mode", obstacle_mode)
            ).strip().lower()
            proposed_omega = obstacle_context.get("proposed_omega")
            avoidance_state = str(obstacle_context.get("avoidance_state", "CLEAR"))
        try:
            proposed_omega_value = float(proposed_omega)
        except (TypeError, ValueError):
            proposed_omega_value = 0.0
        obstacle_in_deadband = (
            scan_result.get("reason") in ("front_obstacle_slow", "front_soft_block")
            or planner_obstacles > 0
        )
        obstacle_turn_mode = bool(
            obstacle_context.get("obstacle_turn_mode", False)
            if obstacle_context
            else False
        )
        if avoidance_state in ("CLEAR", "GOAL_REACQUIRE"):
            obstacle_turn_mode = False
        side_soft_avoid = bool(
            obstacle_context.get("side_soft_avoid", False)
            if obstacle_context
            else False
        )
        goal_reacquire_active = avoidance_state == "GOAL_REACQUIRE"
        goal_tracking_suppression_release_reason = "none"
        if (
            scan_result.get("reason") == "front_clear"
            and planner_obstacles <= 0
            and dynamic_obstacles_current_scan <= 0
        ):
            goal_tracking_suppression_release_reason = "front_clear_no_obstacles"
        elif goal_reacquire_active:
            goal_tracking_suppression_release_reason = "goal_reacquire_state"

        mppi_goal_tracking_sign_guard = (
            str(getattr(self, "planner_mode", "")).strip().lower() == "mppi"
            and sign_of(proposed_omega_value) != 0
        )
        geometric_obstacle_sign_guard = (
            obstacle_mode == "geometric"
            and (dynamic_obstacles_current_scan > 0 or planner_obstacles > 0)
        )
        goal_tracking_sign_guard_active = (
            mppi_goal_tracking_sign_guard or geometric_obstacle_sign_guard
        )
        goal_tracking_reference_omega = omega_before
        if sign_of(goal_tracking_reference_omega) == 0 and sign_of(proposed_omega_value) != 0:
            goal_tracking_reference_omega = proposed_omega_value
        goal_tracking_reverse_blocked = False
        anti_goal_drive_blocked = False
        heading_gate_reason = "none"

        if obstacle_turn_mode or side_soft_avoid or avoidance_state in (
            "APPROACH_SLOW",
            "CREEP_ESCAPE",
            "HARD_STOP_RECOVERY",
        ):
            tracking_mode = (
                "preemptive_obstacle_turn"
                if obstacle_turn_mode
                else "side_soft_avoid"
            )
            if avoidance_state == "CREEP_ESCAPE":
                tracking_mode = "creep_escape_goal_suppressed"
            elif avoidance_state == "APPROACH_SLOW":
                tracking_mode = "approach_slow_goal_suppressed"
            elif avoidance_state == "HARD_STOP_RECOVERY":
                tracking_mode = "hard_stop_goal_suppressed"
            if obstacle_turn_mode:
                omega_limit = float(getattr(config.safety, "front_turn_max_omega", omega_track_max))
            else:
                omega_limit = float(getattr(config.safety, "side_avoid_max_omega", omega_track_max))
            if avoidance_state == "CREEP_ESCAPE":
                omega_limit = float(getattr(config.safety, "planner_creep_max_omega", omega_limit))
            tracking_omega_limit = omega_limit
            omega = clip_value(
                omega_before,
                -omega_limit,
                omega_limit,
            )
            goal_tracking_reason = "suppressed_during_obstacle_avoidance"
        elif abs_yaw_error_deg > yaw_align:
            tracking_mode = "heading_align_soft"
            v_scale = min_v_scale
            v *= v_scale
            anti_goal_drive_blocked = True
            heading_gate_reason = "yaw_align"
            omega_goal = clip_value(k_yaw * yaw_error, -omega_align_max, omega_align_max)
            if (
                sign_of(omega_before) != 0
                and sign_of(omega_goal) != 0
                and sign_of(omega_before) != sign_of(omega_goal)
                and not suppress_wrong_sign
            ):
                omega = omega_before
                goal_tracking_reason = "preserve_mppi_wrong_sign_suppressed"
            else:
                omega = 0.80 * omega_before + 0.20 * omega_goal
                goal_tracking_reason = "weak_heading_align_blend"
        elif abs_yaw_error_deg > yaw_slow:
            tracking_mode = "heading_slow_gate"
            ratio = (yaw_align - abs_yaw_error_deg) / max(
                yaw_align - yaw_slow,
                1e-6,
            )
            v_scale = clip_value(min_v_scale + (1.0 - min_v_scale) * ratio, min_v_scale, 1.0)
            v *= v_scale
            heading_gate_reason = "yaw_slow"
            omega_goal = clip_value(k_yaw * yaw_error, -omega_track_max, omega_track_max)
            if (
                sign_of(omega_before) != 0
                and sign_of(omega_goal) != 0
                and sign_of(omega_before) != sign_of(omega_goal)
                and not suppress_wrong_sign
            ):
                omega = omega_before
                goal_tracking_reason = "preserve_mppi_wrong_sign_suppressed"
            else:
                omega = 0.75 * omega_before + 0.25 * omega_goal
                goal_tracking_reason = "weak_heading_slow_blend"
        elif abs_yaw_error_deg > yaw_deadband:
            heading_gate_reason = "yaw_track"
            omega_goal = clip_value(k_yaw * yaw_error, -omega_track_max, omega_track_max)
            wrong_sign = sign_of(omega_before) != 0 and sign_of(omega_before) != sign_of(yaw_error)
            too_large = abs(omega_before) > max(abs(omega_goal) * 1.35, omega_deadband_max)
            if (suppress_wrong_sign and wrong_sign) or too_large:
                omega = omega_goal
                tracking_mode = "goal_track_override"
                goal_tracking_reason = "bounded_goal_track_override"
            else:
                omega = 0.20 * omega_goal + 0.80 * omega_before
                tracking_mode = "goal_track_blend"
                goal_tracking_reason = "weak_goal_track_blend"
        else:
            if obstacle_in_deadband:
                tracking_mode = "obstacle_yaw_deadband_escape"
                omega = clip_value(omega_before, -0.15, 0.15)
                obstacle_yaw_deadband_escape_applied = True
                goal_tracking_reason = "obstacle_deadband_clamp"
            else:
                tracking_mode = "yaw_deadband"
                if sign_of(omega_before) == sign_of(yaw_error) and sign_of(yaw_error) != 0:
                    omega = clip_value(omega_before, -omega_deadband_max, omega_deadband_max)
                else:
                    omega = 0.0
                smoothing_reset_reason = "yaw_deadband"
                goal_tracking_reason = "yaw_deadband"

        reference_sign = sign_of(goal_tracking_reference_omega)
        if goal_tracking_sign_guard_active and reference_sign != 0:
            omega_sign = sign_of(omega)
            if omega_sign != reference_sign:
                guard_limit = min(float(config.limits.w_max), tracking_omega_limit)
                if abs(float(omega)) > 1e-6:
                    guarded_abs = min(abs(float(omega)), guard_limit)
                else:
                    guarded_abs = min(abs(float(goal_tracking_reference_omega)), guard_limit)
                if guarded_abs > 1e-6:
                    omega = float(reference_sign) * guarded_abs
                    goal_tracking_reverse_blocked = True
                    goal_tracking_reason = "mppi_omega_sign_preserved"

        if reset_on_flip and sign_of(omega_before) != 0 and sign_of(omega) != 0:
            if sign_of(omega_before) != sign_of(omega):
                smoothing_reset_reason = "omega_sign_flip"

        omega = clip_value(
            omega,
            -min(float(config.limits.w_max), tracking_omega_limit),
            min(float(config.limits.w_max), tracking_omega_limit),
        )
        goal_tracking_intervened = (
            abs(float(omega) - omega_before) > 1e-6
            or abs(float(v) - float(control[0])) > 1e-6
        )
        debug = dict(heading_debug)
        debug.update({
            "tracking_mode": tracking_mode,
            "tracking_v_scale": v_scale,
            "omega_before_tracking": omega_before,
            "omega_after_tracking": omega,
            "tracking_omega_before": omega_before,
            "tracking_omega_after": omega,
            "goal_tracking_intervened": goal_tracking_intervened,
            "goal_tracking_reason": goal_tracking_reason,
            "smoothing_reset_reason": smoothing_reset_reason,
            "obstacle_yaw_deadband_escape_applied": obstacle_yaw_deadband_escape_applied,
            "goal_tracking_sign_guard_active": goal_tracking_sign_guard_active,
            "goal_tracking_reference_omega": goal_tracking_reference_omega,
            "goal_tracking_reverse_blocked": goal_tracking_reverse_blocked,
            "goal_reacquire_active": goal_reacquire_active,
            "anti_goal_drive_blocked": anti_goal_drive_blocked,
            "heading_gate_reason": heading_gate_reason,
            "goal_tracking_suppression_release_reason": goal_tracking_suppression_release_reason,
        })
        return (v, omega), debug

    def front_stop_rotate_recovery(self, heading_debug, scan_result):
        self.ensure_runtime_state_defaults()
        def recovery_info(direction="none", left_count=0, right_count=0):
            now_info = time.time()
            if self.hard_stop_recovery_start_time is None:
                age_sec = 0.0
            else:
                age_sec = now_info - float(self.hard_stop_recovery_start_time)
            min_front_range = scan_result.get("min_front_range")
            release_margin = float(
                getattr(self.cfg.safety, "hard_stop_release_margin", 0.04)
            )
            hard_stop_distance = float(
                getattr(self.cfg.safety, "hard_stop_distance", 0.18)
            )
            release_candidate = (
                min_front_range is not None
                and float(min_front_range) > hard_stop_distance + release_margin
            )
            return {
                "recovery_direction": direction,
                "hard_stop_recovery_direction": direction,
                "left_guard_count": left_count,
                "right_guard_count": right_count,
                "front_stop_release_distance": float(
                    getattr(self.cfg.safety, "front_stop_release_distance", 0.36)
                ),
                "hard_stop_recovery_age_sec": age_sec,
                "hard_stop_recovery_switch_reason": "none",
                "hard_stop_release_candidate": bool(release_candidate),
                "hard_stop_exit_to_creep": False,
            }

        if not bool(getattr(self.cfg.safety, "allow_rotate_in_front_stop", True)):
            return (0.0, 0.0), False, recovery_info()

        now = time.time()
        if self.hard_stop_recovery_start_time is None:
            self.hard_stop_recovery_start_time = now
            self.hard_stop_recovery_frames = 0
            self.hard_stop_recovery_no_improve_frames = 0
        self.hard_stop_recovery_frames += 1

        omega_max = float(getattr(self.cfg.safety, "front_stop_recovery_w_max", 0.25))
        omega_min = float(getattr(self.cfg.safety, "front_stop_recovery_w_min", 0.15))
        if omega_max <= 0.0:
            return (0.0, 0.0), False, recovery_info()
        omega_min = clip_value(omega_min, 0.0, omega_max)

        left_guard_count = 0
        right_guard_count = 0
        for point in scan_result.get("front_points", []):
            y_value = float(point.get("y", 0.0))
            if y_value >= 0.0:
                left_guard_count += 1
            else:
                right_guard_count += 1

        left_clearance, right_clearance = self.scan_side_clearance()
        clearance_direction = "none"
        if left_clearance is not None and right_clearance is not None:
            if float(left_clearance) > float(right_clearance) + 0.04:
                clearance_direction = "left"
            elif float(right_clearance) > float(left_clearance) + 0.04:
                clearance_direction = "right"

        if clearance_direction in ("left", "right"):
            direction = clearance_direction
            switch_reason = "side_clearance"
        elif self.avoidance_side in ("left", "right"):
            direction = self.avoidance_side
            switch_reason = "last_corridor_direction"
        elif left_guard_count < right_guard_count:
            direction = "left"
            switch_reason = "front_point_distribution"
        elif right_guard_count < left_guard_count:
            direction = "right"
            switch_reason = "front_point_distribution"
        else:
            yaw_error = float(heading_debug.get("yaw_error", 0.0))
            if abs(yaw_error) > 1e-6:
                direction = "left" if yaw_error > 0.0 else "right"
                switch_reason = "goal_heading_fallback"
            elif self.front_stop_recovery_direction in ("left", "right"):
                direction = self.front_stop_recovery_direction
                switch_reason = "hold_previous"
            else:
                direction = "left"
                switch_reason = "default_left"

        min_front_range = scan_result.get("min_front_range")
        if min_front_range is not None and self.hard_stop_recovery_last_min_front is not None:
            if float(min_front_range) <= float(self.hard_stop_recovery_last_min_front) + 0.005:
                self.hard_stop_recovery_no_improve_frames += 1
            else:
                self.hard_stop_recovery_no_improve_frames = 0
        if min_front_range is not None:
            self.hard_stop_recovery_last_min_front = float(min_front_range)

        if self.front_stop_recovery_direction in ("left", "right"):
            min_switch_sec = float(
                getattr(self.cfg.safety, "hard_stop_recovery_switch_min_sec", 0.8)
            )
            direction_age = now - float(self.hard_stop_recovery_direction_since or now)
            if (
                direction != self.front_stop_recovery_direction
                and direction_age < min_switch_sec
            ):
                direction = self.front_stop_recovery_direction
                switch_reason = "hysteresis_hold"

        no_improve_switch_frames = int(
            getattr(self.cfg.safety, "hard_stop_recovery_no_improve_frames", 8)
        )
        no_improve_switch_frames = max(2, no_improve_switch_frames)
        omega_scale = 1.0
        if self.hard_stop_recovery_no_improve_frames >= no_improve_switch_frames:
            omega_scale = 0.75
        if self.hard_stop_recovery_no_improve_frames >= no_improve_switch_frames * 2:
            min_switch_sec = float(
                getattr(self.cfg.safety, "hard_stop_recovery_switch_min_sec", 0.8)
            )
            direction_age = now - float(self.hard_stop_recovery_direction_since or now)
            if direction_age >= min_switch_sec and self.front_stop_recovery_direction in (
                "left",
                "right",
            ):
                direction = "right" if self.front_stop_recovery_direction == "left" else "left"
                switch_reason = "no_improvement_hysteresis_switch"
                self.hard_stop_recovery_no_improve_frames = 0
                omega_scale = 0.85

        direction_sign = 1.0 if direction == "left" else -1.0
        omega = direction_sign * max(omega_min, omega_max * omega_scale)

        if bool(getattr(self.cfg.safety, "front_stop_recovery_use_goal_direction", True)):
            goal_omega = clip_value(
                float(getattr(self.cfg.tracking, "k_yaw", 0.75))
                * float(heading_debug["yaw_error"]),
                -omega_max,
                omega_max,
            )
            if abs(goal_omega) >= omega_min:
                omega = goal_omega
                direction = "left" if omega > 0.0 else "right"

        if abs(omega) < omega_min:
            omega = (1.0 if omega >= 0.0 else -1.0) * omega_min

        if direction != self.front_stop_recovery_direction:
            self.hard_stop_recovery_direction_since = now
        self.front_stop_recovery_direction = direction
        info = recovery_info(direction, left_guard_count, right_guard_count)
        info["hard_stop_recovery_switch_reason"] = switch_reason
        info["hard_stop_recovery_no_improve_frames"] = self.hard_stop_recovery_no_improve_frames
        info["hard_stop_recovery_omega_scale"] = omega_scale
        info["left_clearance"] = left_clearance
        info["right_clearance"] = right_clearance

        return (0.0, omega), abs(omega) > 1e-6, info

    def side_soft_avoid_would_be_active(self, scan_result):
        min_side_range = scan_result.get("min_side_range")
        side_soft_distance = float(getattr(self.cfg.safety, "side_soft_distance", 0.32))
        side_hard_distance = float(getattr(self.cfg.safety, "side_hard_distance", 0.18))
        side_release_distance = float(getattr(self.cfg.safety, "side_release_distance", 0.36))
        if scan_result.get("reason") == "side_hard_stop":
            return False
        if min_side_range is None:
            return False
        if (
            self.side_avoid_active
            and time.time() < self.side_avoid_until
            and float(min_side_range) < side_release_distance
        ):
            return True
        return (
            side_soft_distance > 0.0
            and float(min_side_range) < side_soft_distance
            and float(min_side_range) >= side_hard_distance
        )

    def update_side_soft_state(self, scan_result):
        now = time.time()
        side = scan_result.get("side_obstacle_side", "none")
        min_side_range = scan_result.get("min_side_range")
        side_release_distance = float(getattr(self.cfg.safety, "side_release_distance", 0.36))
        hold_sec = float(getattr(self.cfg.safety, "side_avoid_hold_sec", 0.8))
        current_active = self.side_soft_avoid_would_be_active(scan_result)

        if current_active:
            self.side_avoid_active = True
            self.side_avoid_until = now + hold_sec
            if side in ("left", "right"):
                self.side_avoid_side = side
        elif (
            self.side_avoid_active
            and now < self.side_avoid_until
            and min_side_range is not None
            and float(min_side_range) < side_release_distance
        ):
            if side in ("left", "right"):
                self.side_avoid_side = side
        else:
            self.side_avoid_active = False
            self.side_avoid_until = 0.0
            self.side_avoid_side = "none"

        return self.side_avoid_active

    def side_avoid_omega_sign(self, side):
        if side == "left":
            return -1.0
        if side == "right":
            return 1.0
        return 0.0

    def apply_side_soft_avoid(self, control, scan_result, obstacle_turn_debug=None):
        omega_before = float(control[1])
        debug = {
            "side_avoid_mode": "none",
            "side_stop_mode": scan_result.get("side_stop_mode", "none"),
            "side_obstacle_side": scan_result.get("side_obstacle_side", "none"),
            "side_soft_distance": float(getattr(self.cfg.safety, "side_soft_distance", 0.32)),
            "side_hard_distance": float(getattr(self.cfg.safety, "side_hard_distance", 0.18)),
            "side_release_distance": float(getattr(self.cfg.safety, "side_release_distance", 0.36)),
            "omega_before_side_avoid": omega_before,
            "omega_after_side_avoid": omega_before,
            "side_soft_min_v": float(getattr(self.cfg.safety, "side_soft_min_v", 0.05)),
            "side_avoid_v_before": float(control[0]),
            "side_avoid_v_after": float(control[0]),
            "side_avoid_applied": False,
            "side_hard_stop_applied": False,
        }

        active = self.update_side_soft_state(scan_result)
        side = self.side_avoid_side
        if not active or side not in ("left", "right"):
            return control, debug

        desired_sign = self.side_avoid_omega_sign(side)
        if obstacle_turn_debug is not None:
            turn_direction = obstacle_turn_debug.get("obstacle_turn_direction", "none")
            if self.direction_pushes_toward_near_side(turn_direction, scan_result):
                desired_sign = self.side_avoid_omega_sign(side)

        min_omega = float(getattr(self.cfg.safety, "side_avoid_min_omega", 0.12))
        max_omega = float(getattr(self.cfg.safety, "side_avoid_max_omega", 0.24))
        v_scale = clip_value(
            getattr(self.cfg.safety, "side_avoid_v_scale", 0.80),
            0.0,
            1.0,
        )
        side_soft_min_v = min(
            float(getattr(self.cfg.safety, "side_soft_min_v", 0.05)),
            float(self.cfg.limits.v_max),
        )
        v_before = float(control[0])
        v = max(v_before * v_scale, side_soft_min_v)
        omega = omega_before
        side_arbitration_mode = "mppi"
        if desired_sign != 0.0:
            omega_sign = sign_of(omega_before, 0.02)
            if omega_sign == 0:
                omega = desired_sign * min(max_omega, max(min_omega, abs(omega_before)))
                side_arbitration_mode = "bias"
            elif omega_sign == sign_of(desired_sign):
                omega = desired_sign * clip_value(abs(omega_before), min_omega, max_omega)
                if abs(omega - omega_before) > 1e-6:
                    side_arbitration_mode = "clamp"
            else:
                omega = omega_sign * min(abs(omega_before), max_omega)
                v = min(v, max(side_soft_min_v, v_before * 0.65))
                side_arbitration_mode = "sign_preserving_clamp"

        debug.update(
            {
                "side_avoid_mode": "side_soft_avoid",
                "side_obstacle_side": side,
                "omega_after_side_avoid": omega,
                "side_avoid_v_after": v,
                "side_avoid_applied": True,
                "side_arbitration_mode": side_arbitration_mode,
            }
        )
        return (v, omega), debug

    def side_hard_stop_recovery(self, scan_result):
        side = scan_result.get("side_obstacle_side", "none")
        min_omega = float(getattr(self.cfg.safety, "side_avoid_min_omega", 0.12))
        max_omega = float(getattr(self.cfg.safety, "side_avoid_max_omega", 0.24))
        if max_omega <= 0.0 or side not in ("left", "right"):
            return (0.0, 0.0), False, {
                "side_avoid_mode": "none",
                "side_stop_mode": "side_hard_stop",
                "side_obstacle_side": side,
            }

        min_omega = clip_value(min_omega, 0.0, max_omega)
        desired_sign = self.side_avoid_omega_sign(side)
        omega = desired_sign * max(min_omega, max_omega)
        return (0.0, omega), True, {
            "side_avoid_mode": "side_hard_recovery",
            "side_stop_mode": "side_hard_stop",
            "side_obstacle_side": side,
            "omega_before_side_avoid": 0.0,
            "omega_after_side_avoid": omega,
            "side_avoid_applied": True,
            "side_hard_stop_applied": True,
        }

    def apply_front_soft_block_creep(self, control, scan_result):
        self.ensure_runtime_state_defaults()
        front_stop_mode = scan_result.get("front_stop_mode", scan_result.get("reason", ""))
        front_delta = self.last_min_front_range_delta
        goal_delta = self.last_goal_distance_delta
        debug = {
            "front_stop_mode": front_stop_mode,
            "planner_creep_mode": False,
            "planner_creep_v": float(getattr(self.cfg.safety, "planner_creep_v", 0.035)),
            "planner_creep_max_omega": float(
                getattr(self.cfg.safety, "planner_creep_max_omega", 0.20)
            ),
            "planner_creep_duration_sec": 0.0,
            "planner_creep_expired": False,
            "creep_escape_active": False,
            "creep_v_selected": float(control[0]),
            "creep_omega_limit": float(
                getattr(self.cfg.safety, "planner_creep_max_omega", 0.20)
            ),
            "creep_escape_reason": "none",
            "creep_progress_ok": False,
            "creep_front_range_delta": front_delta,
            "creep_goal_distance_delta": goal_delta,
        }

        if front_stop_mode != "front_soft_block":
            self.front_soft_block_start_time = None
            return control, debug

        if not bool(getattr(self.cfg.safety, "planner_creep_enable", True)):
            return control, debug

        now = time.time()
        if self.front_soft_block_start_time is None:
            self.front_soft_block_start_time = now
        creep_duration = now - self.front_soft_block_start_time
        creep_max_duration = float(getattr(self.cfg.safety, "creep_max_duration_sec", 2.0))
        debug["planner_creep_duration_sec"] = creep_duration
        debug["planner_creep_expired"] = (
            creep_max_duration > 0.0 and creep_duration > creep_max_duration
        )

        base_creep_v = clip_value(
            getattr(self.cfg.safety, "planner_creep_v", 0.060),
            0.055,
            min(0.070, float(self.cfg.limits.v_max)),
        )
        front_worsening = self.min_front_range_trend == "decreasing"
        creep_progress_ok = self.min_front_range_trend in ("stable", "increasing", "unknown")
        if front_worsening:
            creep_v = max(
                float(getattr(self.cfg.safety, "soft_avoid_min_v", 0.055)),
                base_creep_v * 0.85,
            )
            creep_max_omega = min(
                float(getattr(self.cfg.safety, "front_turn_max_omega", 0.18)),
                float(self.cfg.limits.w_max),
            )
            creep_reason = "front_range_decreasing"
        else:
            creep_v = base_creep_v
            creep_max_omega = min(
                float(getattr(self.cfg.safety, "planner_creep_max_omega", 0.14)),
                0.14,
                float(self.cfg.limits.w_max),
            )
            creep_reason = "front_range_stable_or_opening"
        omega = clip_value(float(control[1]), -creep_max_omega, creep_max_omega)
        debug.update({
            "planner_creep_mode": True,
            "planner_creep_v": creep_v,
            "planner_creep_max_omega": creep_max_omega,
            "creep_escape_active": True,
            "creep_v_selected": creep_v,
            "creep_omega_limit": creep_max_omega,
            "creep_escape_reason": creep_reason,
            "creep_progress_ok": creep_progress_ok,
            "creep_front_range_delta": front_delta,
            "creep_goal_distance_delta": goal_delta,
        })
        return (creep_v, omega), debug

    def apply_soft_avoid_velocity_floor(self, control, scan_result, side_avoid_debug=None):
        v_before = float(control[0])
        omega = float(control[1])
        v_after = v_before
        reason = "none"
        front_slow_min_scale = float(getattr(self.cfg.safety, "front_slow_min_scale", 0.45))
        side_soft_min_v = float(getattr(self.cfg.safety, "side_soft_min_v", 0.05))
        soft_min_v = float(getattr(self.cfg.safety, "soft_avoid_min_v", 0.055))
        soft_target_v = float(getattr(self.cfg.safety, "soft_avoid_target_v", 0.075))
        v_max = float(self.cfg.limits.v_max)

        scan_reason = scan_result.get("reason", "")
        front_stop_mode = scan_result.get("front_stop_mode", scan_reason)
        side_soft_active = (
            scan_reason == "side_obstacle_soft"
            or scan_result.get("side_avoid_mode") == "side_soft_avoid"
            or (
                side_avoid_debug is not None
                and side_avoid_debug.get("side_avoid_mode") == "side_soft_avoid"
            )
        )

        if front_stop_mode == "front_soft_block":
            pass
        elif front_stop_mode == "front_obstacle_slow" or scan_reason == "front_obstacle_slow":
            floor_v = min(max(soft_min_v, soft_target_v), v_max)
            if v_after < floor_v:
                v_after = floor_v
                reason = "front_obstacle_slow"
        elif side_soft_active:
            floor_v = min(side_soft_min_v, v_max)
            if v_after < floor_v:
                v_after = floor_v
                reason = "side_obstacle_soft"

        debug = {
            "soft_avoid_v_before": v_before,
            "soft_avoid_v_after": v_after,
            "soft_avoid_reason": reason,
            "front_slow_min_scale": front_slow_min_scale,
            "side_soft_min_v": side_soft_min_v,
        }
        return (v_after, omega), debug

    def final_safety_clamp(self, control):
        v = clip_value(float(control[0]), 0.0, float(self.cfg.limits.v_max))
        omega = clip_value(
            float(control[1]),
            -float(self.cfg.limits.w_max),
            float(self.cfg.limits.w_max),
        )
        return (v, omega)

    def smooth_control_for_publish(
        self,
        control,
        force_zero=False,
        context="clear",
        goal_tracking_active=False,
        yaw_error_deg=0.0,
    ):
        now = time.time()
        if self.last_control_time is None:
            dt = 1.0 / max(1.0, self.rate_hz)
        else:
            dt = now - self.last_control_time
        dt = clip_value(dt, 1.0 / 100.0, 1.0)

        raw_v = float(control[0])
        raw_omega = float(control[1])

        if force_zero:
            self.last_control_time = now
            self.last_smoothed_v = 0.0
            self.last_smoothed_omega = 0.0
            return (0.0, 0.0), {
                "dt": dt,
                "omega_limited": 0.0,
                "omega_lowpass_alpha": float(getattr(self.cfg.limits, "omega_lowpass_alpha", 0.55)),
                "smoothing_alpha_effective": float(
                    getattr(self.cfg.limits, "omega_lowpass_alpha", 0.55)
                ),
                "omega_slew_rate": float(getattr(self.cfg.limits, "omega_slew_rate", 4.0)),
                "smoothing_context": str(context),
                "smoothing_magnitude_clamped": False,
                "smoother_reset_on_direction_switch": False,
                "clear_goal_tracking_smoothing": False,
            }

        omega_slew_rate = float(getattr(self.cfg.limits, "omega_slew_rate", 4.0))
        alpha = clip_value(getattr(self.cfg.limits, "omega_lowpass_alpha", 0.55), 0.0, 1.0)
        smoothing_context = str(context or "clear")
        clear_goal_tracking_smoothing = (
            smoothing_context == "clear"
            and (
                bool(goal_tracking_active)
                or abs(float(yaw_error_deg)) > 30.0
            )
        )
        clear_goal_tracking_min_omega = float(
            getattr(self.cfg.limits, "clear_goal_tracking_min_omega", 0.05)
        )
        if clear_goal_tracking_smoothing:
            alpha = max(
                alpha,
                clip_value(
                    getattr(self.cfg.limits, "clear_goal_tracking_alpha", 0.35),
                    0.0,
                    1.0,
                ),
            )
            omega_slew_rate = max(
                omega_slew_rate,
                float(getattr(self.cfg.limits, "clear_goal_tracking_slew_rate", 1.0)),
            )
        if smoothing_context in ("obstacle", "creep", "hard_stop_recovery"):
            alpha = max(
                alpha,
                clip_value(
                    getattr(self.cfg.limits, "obstacle_omega_lowpass_alpha", 0.85),
                    0.0,
                    1.0,
                ),
            )
            omega_slew_rate = max(
                omega_slew_rate,
                float(getattr(self.cfg.limits, "obstacle_omega_slew_rate", 2.5)),
            )
        if smoothing_context == "hard_stop_recovery":
            alpha = max(
                alpha,
                clip_value(
                    getattr(self.cfg.limits, "hard_stop_omega_lowpass_alpha", 1.0),
                    0.0,
                    1.0,
                ),
            )
            omega_slew_rate = max(
                omega_slew_rate,
                float(getattr(self.cfg.limits, "hard_stop_omega_slew_rate", 4.0)),
            )
        max_delta = max(0.0, omega_slew_rate * dt)
        omega_limited = clip_value(
            raw_omega,
            self.last_smoothed_omega - max_delta,
            self.last_smoothed_omega + max_delta,
        )
        omega = alpha * omega_limited + (1.0 - alpha) * self.last_smoothed_omega
        magnitude_clamped = False
        if (
            smoothing_context in ("obstacle", "creep", "hard_stop_recovery")
            and abs(raw_omega) >= 0.12
            and abs(omega) < 0.10
        ):
            omega = sign_of(raw_omega) * 0.10
            magnitude_clamped = True
        if (
            clear_goal_tracking_smoothing
            and abs(raw_omega) >= clear_goal_tracking_min_omega
            and abs(omega) < clear_goal_tracking_min_omega
        ):
            omega = sign_of(raw_omega) * min(
                abs(raw_omega),
                clear_goal_tracking_min_omega,
            )
            magnitude_clamped = True

        self.last_control_time = now
        self.last_smoothed_v = raw_v
        self.last_smoothed_omega = omega
        return (raw_v, omega), {
            "dt": dt,
            "omega_limited": omega_limited,
            "omega_lowpass_alpha": alpha,
            "smoothing_alpha_effective": alpha,
            "omega_slew_rate": omega_slew_rate,
            "smoothing_context": smoothing_context,
            "smoothing_magnitude_clamped": magnitude_clamped,
            "smoother_reset_on_direction_switch": False,
            "clear_goal_tracking_smoothing": clear_goal_tracking_smoothing,
        }

    def block_smoothing_sign_flip_if_needed(
        self,
        before_smoothing,
        after_smoothing,
        scan_result,
        obstacle_turn_debug,
        side_avoid_debug,
        front_creep_debug,
        planner_debug,
    ):
        omega_before = float(before_smoothing[1])
        omega_after = float(after_smoothing[1])
        debug = {
            "smoothing_sign_flip_blocked": False,
            "smoothing_omega_before_block": omega_after,
            "smoothing_omega_after_block": omega_after,
            "smoothing_block_reason": "none",
            "smoothing_context": "obstacle" if False else "clear",
            "smoothing_alpha_effective": None,
            "smoothing_magnitude_clamped": False,
            "smoother_reset_on_direction_switch": False,
        }

        front_stop_mode = scan_result.get(
            "front_stop_mode",
            scan_result.get("reason", "none"),
        )
        obstacle_context = (
            bool(obstacle_turn_debug.get("obstacle_turn_mode", False))
            or front_stop_mode in ("front_obstacle_slow", "front_soft_block")
            or bool(front_creep_debug.get("planner_creep_mode", False))
            or bool(side_avoid_debug.get("side_avoid_applied", False))
            or side_avoid_debug.get("side_avoid_mode") == "side_soft_avoid"
            or planner_debug.get("recovery_direction", "none") != "none"
            or bool(obstacle_turn_debug.get("avoidance_direction_locked", False))
            or bool(planner_debug.get("avoidance_direction_locked", False))
        )
        debug["smoothing_context"] = "obstacle" if obstacle_context else "clear"
        sign_flipped = (
            abs(omega_before) > 0.03
            and abs(omega_after) > 0.03
            and omega_before * omega_after < 0.0
        )
        if obstacle_context and sign_flipped:
            after_smoothing = (after_smoothing[0], omega_before)
            self.last_smoothed_omega = omega_before
            debug["smoothing_sign_flip_blocked"] = True
            debug["smoothing_omega_after_block"] = omega_before
            debug["smoothing_block_reason"] = "obstacle_context_sign_preserve"

        return after_smoothing, debug

    def publish_zero_twist(self, reason=""):
        zero_twist = {
            "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        self.reset_control_filter()

        if self.enable_publish and self.cmd_pub is not None:
            self.cmd_pub.publish(make_twist_from_dict(self.twist_type, zero_twist))

        if reason:
            self.rospy.logwarn(
                "[mppi_ros_adapter_skeleton] zero Twist requested: {}".format(reason)
            )

    def _mppi_profile_int(self, name, default_value):
        try:
            return int(getattr(self.cfg.mppi, name, default_value))
        except (TypeError, ValueError):
            return int(default_value)

    def _mppi_profile_float(self, name, default_value):
        try:
            return float(getattr(self.cfg.mppi, name, default_value))
        except (TypeError, ValueError):
            return float(default_value)

    def mppi_runtime_profile_values(self, profile_name):
        base_samples = self._mppi_profile_int("num_samples", 350)
        base_horizon = self._mppi_profile_int("horizon", 25)

        if profile_name == "emergency_degraded":
            samples = self._mppi_profile_int("emergency_degraded_num_samples", 180)
            horizon = self._mppi_profile_int("emergency_degraded_horizon", 20)
        elif profile_name == "realtime_degraded":
            samples = self._mppi_profile_int("realtime_degraded_num_samples", 200)
            horizon = self._mppi_profile_int("realtime_degraded_horizon", 20)
        else:
            profile_name = "normal"
            samples = base_samples
            horizon = base_horizon

        return {
            "runtime_profile": profile_name,
            "num_samples": max(1, int(samples)),
            "horizon": max(1, int(horizon)),
            "base_num_samples": max(1, int(base_samples)),
            "base_horizon": max(1, int(base_horizon)),
        }

    def apply_mppi_runtime_profile(self):
        profile_name = getattr(self, "mppi_runtime_profile", "normal")
        values = self.mppi_runtime_profile_values(profile_name)
        if self.mppi_bridge is not None and hasattr(self.mppi_bridge, "set_runtime_profile"):
            self.mppi_bridge.set_runtime_profile(
                values["runtime_profile"],
                num_samples=values["num_samples"],
                horizon=values["horizon"],
                reason=getattr(self, "mppi_degraded_reason", "normal"),
            )
        return values

    def degraded_level_from_profile(self, profile_name):
        profile_name = str(profile_name or "normal")
        if profile_name == "emergency_degraded":
            return "emergency"
        if profile_name == "realtime_degraded":
            return "realtime"
        return "normal"

    def update_mppi_runtime_profile_after_compute(self, planner_debug, overrun):
        core_ms = planner_debug.get("planner_core_compute_ms")
        if core_ms is None:
            core_ms = planner_debug.get("planner_compute_ms", 0.0)
        try:
            core_ms = float(core_ms)
        except (TypeError, ValueError):
            core_ms = 0.0

        threshold_ms = self._mppi_profile_float("realtime_core_ms_threshold", 180.0)
        recover_ms = self._mppi_profile_float("realtime_recover_core_ms", 140.0)
        emergency_streak = self._mppi_profile_int("emergency_degraded_core_streak", 4)
        recover_streak_target = self._mppi_profile_int("realtime_recover_streak", 5)
        previous_profile = getattr(self, "mppi_runtime_profile", "normal")
        degraded_enter_reason = "none"
        degraded_exit_reason = "none"

        if core_ms > threshold_ms:
            self.mppi_core_overrun_streak = getattr(self, "mppi_core_overrun_streak", 0) + 1
            self.mppi_core_recover_streak = 0
            if self.mppi_core_overrun_streak >= max(1, emergency_streak):
                self.mppi_runtime_profile = "emergency_degraded"
                self.mppi_degraded_reason = "core_ms_over_threshold_streak_{}".format(
                    self.mppi_core_overrun_streak
                )
            else:
                self.mppi_runtime_profile = "realtime_degraded"
                self.mppi_degraded_reason = "core_ms_over_threshold_{:.1f}".format(core_ms)
            if self.mppi_runtime_profile != previous_profile:
                degraded_enter_reason = self.mppi_degraded_reason
        elif core_ms <= recover_ms and not overrun:
            self.mppi_core_overrun_streak = max(
                0,
                getattr(self, "mppi_core_overrun_streak", 0) - 1,
            )
            self.mppi_core_recover_streak = getattr(self, "mppi_core_recover_streak", 0) + 1
            if self.mppi_core_recover_streak >= max(1, recover_streak_target):
                if getattr(self, "mppi_runtime_profile", "normal") == "emergency_degraded":
                    self.mppi_runtime_profile = "realtime_degraded"
                    self.mppi_degraded_reason = "recovering_from_emergency"
                    self.mppi_core_recover_streak = 0
                    degraded_exit_reason = "emergency_core_recovered"
                else:
                    self.mppi_runtime_profile = "normal"
                    self.mppi_degraded_reason = "normal"
                    degraded_exit_reason = "core_recover_streak"
        else:
            self.mppi_core_recover_streak = 0

        planner_debug["next_runtime_profile"] = getattr(
            self,
            "mppi_runtime_profile",
            "normal",
        )
        planner_debug["next_degraded_reason"] = getattr(
            self,
            "mppi_degraded_reason",
            "normal",
        )
        planner_debug["mppi_core_overrun_streak"] = getattr(
            self,
            "mppi_core_overrun_streak",
            0,
        )
        planner_debug["core_overrun_streak"] = getattr(
            self,
            "mppi_core_overrun_streak",
            0,
        )
        planner_debug["core_recover_streak"] = getattr(
            self,
            "mppi_core_recover_streak",
            0,
        )
        planner_debug["degraded_level"] = self.degraded_level_from_profile(
            getattr(self, "mppi_runtime_profile", "normal")
        )
        planner_debug["degraded_enter_reason"] = degraded_enter_reason
        planner_debug["degraded_exit_reason"] = degraded_exit_reason
        planner_debug["planner_skipped_no_core_update"] = False

    def maybe_publish_control(self, twist_dict):
        if not self.enable_publish or self.cmd_pub is None:
            return

        self.cmd_pub.publish(make_twist_from_dict(self.twist_type, twist_dict))

    def should_log_verbose_cycle_status(self, planner_debug, scan_result, overrun, note):
        count = getattr(self, "cycle_status_count", 0) + 1
        self.cycle_status_count = count

        signature = (
            scan_result.get("reason"),
            planner_debug.get(
                "front_stop_mode",
                scan_result.get("front_stop_mode", scan_result.get("reason", "none")),
            ),
            planner_debug.get("planner_skip_reason", "none"),
            planner_debug.get("omega_source", "mppi"),
            planner_debug.get("omega_override_reason", "none"),
            planner_debug.get("goal_tracking_reason", "none"),
            bool(planner_debug.get("goal_tracking_reverse_blocked", False)),
            str(note),
        )
        last_signature = getattr(self, "last_cycle_status_signature", None)
        self.last_cycle_status_signature = signature

        if count <= 3 or signature != last_signature:
            return True

        if bool(getattr(self, "realtime_degraded", False)):
            verbose_every = 30
        elif overrun:
            verbose_every = 10
        else:
            verbose_every = 15

        return count % verbose_every == 0

    def print_cycle_status(
        self,
        state,
        goal,
        proposed_control,
        scan_result,
        final_control,
        note,
        planner_debug=None,
    ):
        if state is None:
            state_text = "None"
        else:
            state_text = "({:.3f}, {:.3f}, {:.3f})".format(
                state[0], state[1], state[2]
            )

        if proposed_control is None:
            proposed_text = "None"
        else:
            proposed_text = "({:.3f}, {:.3f})".format(
                proposed_control[0],
                proposed_control[1],
            )

        if final_control is None:
            final_text = "None"
        else:
            final_text = "({:.3f}, {:.3f})".format(final_control[0], final_control[1])

        raw_final_control = None
        if planner_debug is not None:
            raw_final_control = planner_debug.get("raw_final_control")
        if raw_final_control is None:
            raw_final_text = "None"
        else:
            raw_final_text = "({:.3f}, {:.3f})".format(
                raw_final_control[0],
                raw_final_control[1],
            )

        min_range = scan_result.get("min_front_range")
        if min_range is None:
            min_range_text = "None"
        else:
            min_range_text = "{:.3f}".format(min_range)

        min_side_range = scan_result.get("min_side_range")
        if min_side_range is None:
            min_side_range_text = "None"
        else:
            min_side_range_text = "{:.3f}".format(min_side_range)

        if planner_debug is None:
            planner_debug = {}

        planner_type = planner_debug.get("planner_type", self.planner_mode)
        planner_obstacle_count = planner_debug.get("obstacle_count")
        if planner_obstacle_count is None:
            planner_obstacle_count = getattr(self, "latest_dynamic_obstacle_count", None)
        if planner_obstacle_count is None:
            planner_obstacle_text = "None"
        else:
            planner_obstacle_text = str(planner_obstacle_count)

        control_period_target_ms = 1000.0 / max(1.0, float(getattr(self, "rate_hz", 5.0)))
        loop_elapsed_ms = 0.0
        if getattr(self, "current_loop_start_time", None) is not None:
            loop_elapsed_ms = (time.time() - self.current_loop_start_time) * 1000.0
        loop_dt_ms = getattr(self, "current_loop_dt_ms", None)
        if loop_dt_ms is None:
            loop_dt_ms = control_period_target_ms
        if loop_dt_ms is None or loop_dt_ms <= 1e-6:
            achieved_rate_hz = 0.0
        else:
            achieved_rate_hz = 1000.0 / float(loop_dt_ms)
        planner_compute_ms = planner_debug.get("planner_compute_ms")
        if planner_compute_ms is None and planner_debug.get("plan_time_sec") is not None:
            planner_compute_ms = float(planner_debug.get("plan_time_sec")) * 1000.0
        if planner_compute_ms is None:
            planner_compute_ms = 0.0
        planner_compute_ms = float(planner_compute_ms)
        overrun = (
            loop_elapsed_ms > control_period_target_ms
            or planner_compute_ms > control_period_target_ms * 0.85
            or float(loop_dt_ms) > control_period_target_ms * 1.25
        )
        if overrun:
            self.realtime_overrun_streak = getattr(self, "realtime_overrun_streak", 0) + 1
        else:
            self.realtime_overrun_streak = max(
                0,
                getattr(self, "realtime_overrun_streak", 0) - 1,
            )
        if self.realtime_overrun_streak >= 3:
            self.realtime_degraded = True
        elif self.realtime_overrun_streak == 0:
            self.realtime_degraded = False
        planner_debug["loop_dt_ms"] = loop_dt_ms
        planner_debug["planner_compute_ms"] = planner_compute_ms
        planner_debug["achieved_rate_hz"] = achieved_rate_hz
        planner_debug["control_period_target_ms"] = control_period_target_ms
        planner_debug["overrun"] = overrun
        planner_debug["realtime_degraded"] = bool(getattr(self, "realtime_degraded", False))
        planner_skipped_no_core_update = planner_debug.get("planner_type") != "mppi"
        planner_debug["planner_skipped_no_core_update"] = bool(planner_skipped_no_core_update)
        if planner_debug.get("planner_type") == "mppi":
            self.update_mppi_runtime_profile_after_compute(planner_debug, overrun)
        else:
            planner_debug["degraded_reason"] = planner_debug.get(
                "degraded_reason",
                "planner_not_called_no_core_update",
            )
            planner_debug["degraded_level"] = self.degraded_level_from_profile(
                getattr(self, "mppi_runtime_profile", "normal")
            )
            planner_debug["degraded_enter_reason"] = "none"
            planner_debug["degraded_exit_reason"] = "none"
            planner_debug["core_overrun_streak"] = getattr(
                self,
                "mppi_core_overrun_streak",
                0,
            )
            planner_debug["core_recover_streak"] = getattr(
                self,
                "mppi_core_recover_streak",
                0,
            )

        obstacle_debug = self.latest_dynamic_obstacle_debug or {}
        obstacle_mode = obstacle_debug.get(
            "mode",
            getattr(self.cfg.mppi, "scan_obstacle_mode", "raw"),
        )
        obstacle_segment_count = obstacle_debug.get("segment_count", 0)
        obstacle_edge_count = obstacle_debug.get("edge_count", 0)
        obstacle_line_count = obstacle_debug.get("line_surface_count", 0)
        obstacle_circle_count = obstacle_debug.get("circle_obstacle_count", 0)
        obstacle_corner_count = obstacle_debug.get("edge_corner_count", 0)
        obstacle_irregular_count = obstacle_debug.get("irregular_count", 0)
        compressed_line_obstacles = obstacle_debug.get("compressed_line_obstacles", 0)
        line_fit_candidates = obstacle_debug.get("line_fit_candidates", 0)
        line_fit_accepted = obstacle_debug.get("line_fit_accepted", 0)
        line_fit_rejected_reason = obstacle_debug.get("line_fit_rejected_reason", "none")
        line_surface_streak = obstacle_debug.get("line_surface_streak", 0)
        representative_circle_count = obstacle_debug.get(
            "representative_circle_count",
            compressed_line_obstacles,
        )
        obstacle_budget_mode = obstacle_debug.get("obstacle_budget_mode", "bounded")
        geometric_obstacle_count_raw = obstacle_debug.get(
            "geometric_obstacle_count_raw",
            obstacle_debug.get("obstacle_count_before_limit", 0),
        )
        geometric_obstacle_count_used = obstacle_debug.get(
            "geometric_obstacle_count_used",
            obstacle_debug.get("obstacle_count_after_limit", 0),
        )
        line_compression_mode = obstacle_debug.get("line_compression_mode", "none")
        yaw_error_deg = planner_debug.get("yaw_error_deg")
        if yaw_error_deg is None:
            yaw_error_text = "None"
        else:
            yaw_error_text = "{:.1f}".format(float(yaw_error_deg))
        goal_distance = planner_debug.get("goal_distance")
        if goal_distance is None:
            goal_distance_text = "None"
        else:
            goal_distance_text = "{:.3f}".format(float(goal_distance))
        goal_bearing = planner_debug.get("goal_bearing")
        if goal_bearing is None:
            goal_bearing_text = "None"
        else:
            goal_bearing_text = "{:.3f}".format(float(goal_bearing))
        front_corridor_min_range = planner_debug.get("front_corridor_min_range")
        if front_corridor_min_range is None:
            front_corridor_min_range_text = "None"
        else:
            front_corridor_min_range_text = "{:.3f}".format(
                float(front_corridor_min_range)
            )

        planner_text = str(planner_type)
        if planner_type == "mppi":
            best_cost = planner_debug.get("best_cost")
            if best_cost is None:
                best_cost_text = "None"
            else:
                best_cost_text = "{:.3f}".format(float(best_cost))
            planner_text = (
                "mppi,safety_intervened={},best_cost={},samples={}".format(
                    planner_debug.get("safety_intervened"),
                    best_cost_text,
                    planner_debug.get("num_samples"),
                )
            )

        if not self.should_log_verbose_cycle_status(
            planner_debug,
            scan_result,
            overrun,
            note,
        ):
            self.rospy.loginfo(
                "[mppi_ros_adapter_skeleton] summary | "
                "planner={planner} | loop_dt_ms={loop_dt_ms:.1f} | "
                "planner_compute_ms={planner_compute_ms:.1f} | "
                "planner_core_compute_ms={planner_core_compute_ms} | "
                "effective_num_samples={effective_num_samples} | "
                "effective_horizon={effective_horizon} | "
                "degraded_reason={degraded_reason} | "
                "achieved_rate_hz={achieved_rate_hz:.2f} | "
                "control_period_target_ms={control_period_target_ms:.1f} | "
                "overrun={overrun} | realtime_degraded={realtime_degraded} | "
                "degraded_level={degraded_level} | "
                "core_overrun_streak={core_overrun_streak} | "
                "core_recover_streak={core_recover_streak} | "
                "planner_skipped_no_core_update={planner_skipped_no_core_update} | "
                "proposed_control={proposed_control} | final_control={final_control} | "
                "scan_reason={reason} | min_front_range={min_range} | "
                "min_side_range={min_side_range} | front_stop_mode={front_stop_mode} | "
                "avoidance_state={avoidance_state} | "
                "avoidance_state_reason={avoidance_state_reason} | "
                "planner_creep_mode={planner_creep_mode} | "
                "planner_called_under_front_block={planner_called_under_front_block} | "
                "creep_escape_active={creep_escape_active} | "
                "creep_v_selected={creep_v_selected} | "
                "creep_omega_limit={creep_omega_limit} | "
                "soft_avoid_v_before={soft_avoid_v_before} | "
                "soft_avoid_v_after={soft_avoid_v_after} | "
                "omega_source={omega_source} | "
                "arbitration_mode={arbitration_mode} | "
                "hard_override_allowed={hard_override_allowed} | "
                "hard_override_gate_reason={hard_override_gate_reason} | "
                "mppi_direction_conflict={mppi_direction_conflict} | "
                "mppi_conflict_frames={mppi_conflict_frames} | "
                "mppi_omega_preserved={mppi_omega_preserved} | "
                "mppi_omega_overridden={mppi_omega_overridden} | "
                "omega_override_reason={omega_override_reason} | "
                "proposed_omega={proposed_omega} | "
                "omega_before_tracking={omega_before_tracking} | "
                "omega_after_tracking={omega_after_tracking} | "
                "final_omega_before_smoothing={final_omega_before_smoothing} | "
                "final_omega_after_smoothing={final_omega_after_smoothing} | "
                "smoothing_sign_flip_blocked={smoothing_sign_flip_blocked} | "
                "smoothing_context={smoothing_context} | "
                "smoothing_alpha_effective={smoothing_alpha_effective} | "
                "smoothing_magnitude_clamped={smoothing_magnitude_clamped} | "
                "smoother_reset_on_direction_switch={smoother_reset_on_direction_switch} | "
                "smoothing_omega_before_block={smoothing_omega_before_block} | "
                "smoothing_omega_after_block={smoothing_omega_after_block} | "
                "smoothing_block_reason={smoothing_block_reason} | "
                "goal_tracking_reason={goal_tracking_reason} | "
                "goal_reacquire_active={goal_reacquire_active} | "
                "anti_goal_drive_blocked={anti_goal_drive_blocked} | "
                "heading_gate_reason={heading_gate_reason} | "
                "dynamic_obstacles_current_scan={dynamic_obstacles_current_scan} | "
                "planner_obstacles={planner_obstacles} | segments={segments} | "
                "line_surfaces={line_surfaces} | "
                "compressed_line_obstacles={compressed_line_obstacles} | "
                "line_fit_candidates={line_fit_candidates} | "
                "line_fit_accepted={line_fit_accepted} | "
                "line_surface_streak={line_surface_streak} | "
                "representative_circle_count={representative_circle_count} | "
                "goal_distance={goal_distance} | note={note}".format(
                    planner=planner_text,
                    loop_dt_ms=float(loop_dt_ms),
                    planner_compute_ms=planner_compute_ms,
                    planner_core_compute_ms=planner_debug.get(
                        "planner_core_compute_ms",
                        "None",
                    ),
                    effective_num_samples=planner_debug.get(
                        "effective_num_samples",
                        planner_debug.get("num_samples", "None"),
                    ),
                    effective_horizon=planner_debug.get(
                        "effective_horizon",
                        planner_debug.get("horizon", "None"),
                    ),
                    degraded_reason=planner_debug.get(
                        "degraded_reason",
                        getattr(self, "mppi_degraded_reason", "normal"),
                    ),
                    achieved_rate_hz=achieved_rate_hz,
                    control_period_target_ms=control_period_target_ms,
                    overrun=overrun,
                    realtime_degraded=planner_debug.get("realtime_degraded", False),
                    degraded_level=planner_debug.get("degraded_level", "normal"),
                    core_overrun_streak=planner_debug.get("core_overrun_streak", 0),
                    core_recover_streak=planner_debug.get("core_recover_streak", 0),
                    planner_skipped_no_core_update=planner_debug.get(
                        "planner_skipped_no_core_update",
                        False,
                    ),
                    proposed_control=proposed_text,
                    final_control=final_text,
                    reason=scan_result.get("reason"),
                    min_range=min_range_text,
                    min_side_range=min_side_range_text,
                    front_stop_mode=planner_debug.get(
                        "front_stop_mode",
                        scan_result.get("front_stop_mode", scan_result.get("reason", "none")),
                    ),
                    avoidance_state=planner_debug.get("avoidance_state", "CLEAR"),
                    avoidance_state_reason=planner_debug.get(
                        "avoidance_state_reason",
                        "none",
                    ),
                    planner_creep_mode=planner_debug.get("planner_creep_mode", False),
                    planner_called_under_front_block=planner_debug.get(
                        "planner_called_under_front_block",
                        False,
                    ),
                    creep_escape_active=planner_debug.get("creep_escape_active", False),
                    creep_v_selected=planner_debug.get("creep_v_selected", "None"),
                    creep_omega_limit=planner_debug.get("creep_omega_limit", "None"),
                    soft_avoid_v_before=planner_debug.get("soft_avoid_v_before", "None"),
                    soft_avoid_v_after=planner_debug.get("soft_avoid_v_after", "None"),
                    omega_source=planner_debug.get("omega_source", "mppi"),
                    arbitration_mode=planner_debug.get("arbitration_mode", "mppi"),
                    hard_override_allowed=planner_debug.get("hard_override_allowed", False),
                    hard_override_gate_reason=planner_debug.get(
                        "hard_override_gate_reason",
                        "none",
                    ),
                    mppi_direction_conflict=planner_debug.get(
                        "mppi_direction_conflict",
                        False,
                    ),
                    mppi_conflict_frames=planner_debug.get("mppi_conflict_frames", 0),
                    mppi_omega_preserved=planner_debug.get("mppi_omega_preserved", True),
                    mppi_omega_overridden=planner_debug.get("mppi_omega_overridden", False),
                    omega_override_reason=planner_debug.get("omega_override_reason", "none"),
                    proposed_omega=planner_debug.get("proposed_omega", "None"),
                    omega_before_tracking=planner_debug.get("omega_before_tracking", "None"),
                    omega_after_tracking=planner_debug.get("omega_after_tracking", "None"),
                    final_omega_before_smoothing=planner_debug.get(
                        "final_omega_before_smoothing",
                        "None",
                    ),
                    final_omega_after_smoothing=planner_debug.get(
                        "final_omega_after_smoothing",
                        "None",
                    ),
                    smoothing_sign_flip_blocked=planner_debug.get(
                        "smoothing_sign_flip_blocked",
                        False,
                    ),
                    smoothing_context=planner_debug.get("smoothing_context", "clear"),
                    smoothing_alpha_effective=planner_debug.get(
                        "smoothing_alpha_effective",
                        "None",
                    ),
                    smoothing_magnitude_clamped=planner_debug.get(
                        "smoothing_magnitude_clamped",
                        False,
                    ),
                    smoother_reset_on_direction_switch=planner_debug.get(
                        "smoother_reset_on_direction_switch",
                        False,
                    ),
                    smoothing_omega_before_block=planner_debug.get(
                        "smoothing_omega_before_block",
                        "None",
                    ),
                    smoothing_omega_after_block=planner_debug.get(
                        "smoothing_omega_after_block",
                        "None",
                    ),
                    smoothing_block_reason=planner_debug.get(
                        "smoothing_block_reason",
                        "none",
                    ),
                    goal_tracking_reason=planner_debug.get("goal_tracking_reason", "none"),
                    goal_reacquire_active=planner_debug.get("goal_reacquire_active", False),
                    anti_goal_drive_blocked=planner_debug.get("anti_goal_drive_blocked", False),
                    heading_gate_reason=planner_debug.get("heading_gate_reason", "none"),
                    dynamic_obstacles_current_scan=self.latest_dynamic_obstacle_current_scan_count,
                    planner_obstacles=planner_obstacle_text,
                    segments=obstacle_segment_count,
                    line_surfaces=obstacle_line_count,
                    compressed_line_obstacles=compressed_line_obstacles,
                    line_fit_candidates=line_fit_candidates,
                    line_fit_accepted=line_fit_accepted,
                    line_surface_streak=line_surface_streak,
                    representative_circle_count=representative_circle_count,
                    goal_distance=goal_distance_text,
                    note=note,
                )
            )
            return

        self.rospy.loginfo(
            "[mppi_ros_adapter_skeleton] state={state} | "
            "goal=({goal_x:.3f}, {goal_y:.3f}) | planner={planner} | "
            "loop_dt_ms={loop_dt_ms:.1f} | planner_compute_ms={planner_compute_ms:.1f} | "
            "planner_core_compute_ms={planner_core_compute_ms} | "
            "effective_num_samples={effective_num_samples} | "
            "effective_horizon={effective_horizon} | "
            "degraded_reason={degraded_reason} | "
            "achieved_rate_hz={achieved_rate_hz:.2f} | "
            "control_period_target_ms={control_period_target_ms:.1f} | "
            "overrun={overrun} | realtime_degraded={realtime_degraded} | "
            "proposed_control={proposed_control} | "
            "scan_reason={reason} | emergency_stop={emergency_stop} | "
            "front_stop_mode={front_stop_mode} | planner_creep_mode={planner_creep_mode} | "
            "planner_called_under_front_block={planner_called_under_front_block} | "
            "planner_skip_reason={planner_skip_reason} | "
            "slow_scale={slow_scale:.3f} | min_front_range={min_range} | "
            "min_side_range={min_side_range} | front_points_count={front_points_count} | "
            "front_stop_distance={front_stop_distance} | "
            "front_soft_block_distance={front_soft_block_distance} | "
            "hard_stop_distance={hard_stop_distance} | "
            "side_points_count={side_points_count} | scan_frame={scan_frame} | "
            "side_avoid_mode={side_avoid_mode} | side_stop_mode={side_stop_mode} | "
            "side_obstacle_side={side_obstacle_side} | "
            "side_soft_distance={side_soft_distance} | "
            "side_hard_distance={side_hard_distance} | "
            "side_release_distance={side_release_distance} | "
            "omega_before_side_avoid={omega_before_side_avoid} | "
            "omega_after_side_avoid={omega_after_side_avoid} | "
            "side_avoid_applied={side_avoid_applied} | "
            "side_hard_stop_applied={side_hard_stop_applied} | "
            "soft_avoid_v_before={soft_avoid_v_before} | "
            "soft_avoid_v_after={soft_avoid_v_after} | "
            "soft_avoid_reason={soft_avoid_reason} | "
            "front_slow_min_scale={front_slow_min_scale} | "
            "side_soft_min_v={side_soft_min_v} | "
            "base_frame={base_frame} | scan_offset_deg={scan_offset_deg:.1f} | "
            "scan_tf_ok={scan_tf_ok} | "
            "goal_distance={goal_distance} | goal_bearing={goal_bearing} | "
            "yaw_error_deg={yaw_error_deg} | "
            "tracking_mode={tracking_mode} | tracking_v_scale={tracking_v_scale:.3f} | "
            "omega_before_tracking={omega_before_tracking} | "
            "omega_after_tracking={omega_after_tracking} | "
            "tracking_omega_before={tracking_omega_before} | "
            "tracking_omega_after={tracking_omega_after} | "
            "goal_tracking_intervened={goal_tracking_intervened} | "
            "goal_tracking_reason={goal_tracking_reason} | "
            "smoothing_reset_reason={smoothing_reset_reason} | "
            "heading_gate_active={heading_gate_active} | "
            "obstacle_mode={obstacle_mode} | segments={segments} | "
            "use_anisotropic_sampling={use_anisotropic_sampling} | "
            "sdf_influence_distance={sdf_influence_distance} | "
            "scan_obstacle_mode={scan_obstacle_mode} | "
            "edges={edges} | line_surfaces={line_surfaces} | "
            "circle_obstacles={circle_obstacles} | edge_corners={edge_corners} | "
            "irregular={irregular} | "
            "dynamic_obstacles={dynamic_obstacles} | "
            "dynamic_obstacles_current_scan={dynamic_obstacles_current_scan} | "
            "planner_obstacles={planner_obstacles} | "
            "geometric_obstacle_count_raw={geometric_obstacle_count_raw} | "
            "geometric_obstacle_count_used={geometric_obstacle_count_used} | "
            "compressed_line_obstacles={compressed_line_obstacles} | "
            "line_compression_mode={line_compression_mode} | "
            "front_corridor_obstacle_count={front_corridor_obstacle_count} | "
            "front_corridor_min_range={front_corridor_min_range} | "
            "obstacle_turn_mode={obstacle_turn_mode} | "
            "preemptive_turn_candidate={preemptive_turn_candidate} | "
            "obstacle_context_gate_reason={obstacle_context_gate_reason} | "
            "front_turn_distance={front_turn_distance} | "
            "front_turn_release_distance={front_turn_release_distance} | "
            "obstacle_turn_entry_reason={obstacle_turn_entry_reason} | "
            "obstacle_turn_release_reason={obstacle_turn_release_reason} | "
            "obstacle_turn_yaw_damped={obstacle_turn_yaw_damped} | "
            "obstacle_turn_yaw_released={obstacle_turn_yaw_released} | "
            "obstacle_turn_direction={obstacle_turn_direction} | "
            "obstacle_turn_omega={obstacle_turn_omega} | "
            "omega_before_obstacle_turn={omega_before_obstacle_turn} | "
            "omega_after_obstacle_turn={omega_after_obstacle_turn} | "
            "omega_source={omega_source} | "
            "mppi_omega_preserved={mppi_omega_preserved} | "
            "mppi_omega_overridden={mppi_omega_overridden} | "
            "omega_override_reason={omega_override_reason} | "
            "proposed_omega={proposed_omega} | "
            "final_omega_before_smoothing={final_omega_before_smoothing} | "
            "final_omega_after_smoothing={final_omega_after_smoothing} | "
            "smoothing_sign_flip_blocked={smoothing_sign_flip_blocked} | "
            "smoothing_omega_before_block={smoothing_omega_before_block} | "
            "smoothing_omega_after_block={smoothing_omega_after_block} | "
            "smoothing_block_reason={smoothing_block_reason} | "
            "mppi_turn_direction={mppi_turn_direction} | "
            "override_risk_score={override_risk_score} | "
            "turn_v_scale={turn_v_scale} | "
            "avoidance_direction_locked={avoidance_direction_locked} | "
            "avoidance_release_reason={avoidance_release_reason} | "
            "left_corridor_score={left_corridor_score} | "
            "right_corridor_score={right_corridor_score} | "
            "corridor_selected_direction={corridor_selected_direction} | "
            "corridor_switch_allowed={corridor_switch_allowed} | "
            "corridor_switch_reason={corridor_switch_reason} | "
            "corridor_switch_margin={corridor_switch_margin} | "
            "corridor_switch_streak={corridor_switch_streak} | "
            "switch_blocked_by_lock={switch_blocked_by_lock} | "
            "switch_block_reason={switch_block_reason} | "
            "selected_direction_age_sec={selected_direction_age_sec} | "
            "arbitration_mode={arbitration_mode} | "
            "hard_override_allowed={hard_override_allowed} | "
            "hard_override_gate_reason={hard_override_gate_reason} | "
            "mppi_direction_conflict={mppi_direction_conflict} | "
            "mppi_conflict_frames={mppi_conflict_frames} | "
            "min_front_range_trend={min_front_range_trend} | "
            "goal_distance_trend={goal_distance_trend} | "
            "avoidance_state={avoidance_state} | "
            "avoidance_state_reason={avoidance_state_reason} | "
            "avoidance_state_age_sec={avoidance_state_age_sec} | "
            "avoidance_release_candidate={avoidance_release_candidate} | "
            "avoidance_release_block_reason={avoidance_release_block_reason} | "
            "clear_streak={clear_streak} | progress_streak={progress_streak} | "
            "stuck_streak={stuck_streak} | "
            "creep_escape_active={creep_escape_active} | "
            "creep_v_selected={creep_v_selected} | "
            "creep_omega_limit={creep_omega_limit} | "
            "creep_escape_reason={creep_escape_reason} | "
            "creep_progress_ok={creep_progress_ok} | "
            "creep_front_range_delta={creep_front_range_delta} | "
            "creep_goal_distance_delta={creep_goal_distance_delta} | "
            "front_stop_release_distance={front_stop_release_distance} | "
            "recovery_direction={recovery_direction} | "
            "hard_stop_recovery_age_sec={hard_stop_recovery_age_sec} | "
            "hard_stop_recovery_direction={hard_stop_recovery_direction} | "
            "hard_stop_recovery_switch_reason={hard_stop_recovery_switch_reason} | "
            "hard_stop_release_candidate={hard_stop_release_candidate} | "
            "hard_stop_exit_to_creep={hard_stop_exit_to_creep} | "
            "left_guard_count={left_guard_count} | right_guard_count={right_guard_count} | "
            "smoothing_context={smoothing_context} | "
            "smoothing_alpha_effective={smoothing_alpha_effective} | "
            "smoothing_magnitude_clamped={smoothing_magnitude_clamped} | "
            "smoother_reset_on_direction_switch={smoother_reset_on_direction_switch} | "
            "goal_reacquire_active={goal_reacquire_active} | "
            "anti_goal_drive_blocked={anti_goal_drive_blocked} | "
            "heading_gate_reason={heading_gate_reason} | "
            "goal_tracking_suppression_release_reason={goal_tracking_suppression_release_reason} | "
            "degraded_level={degraded_level} | "
            "degraded_enter_reason={degraded_enter_reason} | "
            "degraded_exit_reason={degraded_exit_reason} | "
            "core_overrun_streak={core_overrun_streak} | "
            "core_recover_streak={core_recover_streak} | "
            "planner_skipped_no_core_update={planner_skipped_no_core_update} | "
            "line_fit_candidates={line_fit_candidates} | "
            "line_fit_accepted={line_fit_accepted} | "
            "line_fit_rejected_reason={line_fit_rejected_reason} | "
            "line_surface_streak={line_surface_streak} | "
            "representative_circle_count={representative_circle_count} | "
            "obstacle_budget_mode={obstacle_budget_mode} | "
            "obstacle_yaw_deadband_escape_applied={obstacle_yaw_deadband_escape_applied} | "
            "avoidance_side={avoidance_side} | "
            "avoidance_override_applied={avoidance_override_applied} | "
            "avoidance_hold_remaining={avoidance_hold_remaining:.3f} | "
            "raw_final_control={raw_final_control} | "
            "smoothed_control={smoothed_control} | "
            "final_control={final_control} | publish_enabled={publish_enabled} | "
            "note={note}".format(
                state=state_text,
                goal_x=goal[0],
                goal_y=goal[1],
                planner=planner_text,
                loop_dt_ms=float(loop_dt_ms),
                planner_compute_ms=planner_compute_ms,
                planner_core_compute_ms=planner_debug.get(
                    "planner_core_compute_ms",
                    "None",
                ),
                effective_num_samples=planner_debug.get(
                    "effective_num_samples",
                    planner_debug.get("num_samples", "None"),
                ),
                effective_horizon=planner_debug.get(
                    "effective_horizon",
                    planner_debug.get("horizon", "None"),
                ),
                degraded_reason=planner_debug.get(
                    "degraded_reason",
                    getattr(self, "mppi_degraded_reason", "normal"),
                ),
                achieved_rate_hz=achieved_rate_hz,
                control_period_target_ms=control_period_target_ms,
                overrun=overrun,
                realtime_degraded=planner_debug.get("realtime_degraded", False),
                proposed_control=proposed_text,
                reason=scan_result.get("reason"),
                emergency_stop=scan_result.get("emergency_stop"),
                front_stop_mode=planner_debug.get(
                    "front_stop_mode",
                    scan_result.get("front_stop_mode", scan_result.get("reason", "none")),
                ),
                planner_creep_mode=planner_debug.get("planner_creep_mode", False),
                planner_called_under_front_block=planner_debug.get(
                    "planner_called_under_front_block",
                    False,
                ),
                planner_skip_reason=planner_debug.get("planner_skip_reason", "none"),
                slow_scale=float(scan_result.get("slow_scale", 0.0)),
                min_range=min_range_text,
                min_side_range=min_side_range_text,
                front_points_count=scan_result.get("valid_front_count", 0),
                front_stop_distance=planner_debug.get(
                    "front_stop_distance",
                    scan_result.get(
                        "front_stop_distance",
                        getattr(self.cfg.safety, "front_stop_distance", "None"),
                    ),
                ),
                front_soft_block_distance=planner_debug.get(
                    "front_soft_block_distance",
                    scan_result.get(
                        "front_soft_block_distance",
                        getattr(self.cfg.safety, "front_soft_block_distance", "None"),
                    ),
                ),
                hard_stop_distance=planner_debug.get(
                    "hard_stop_distance",
                    scan_result.get(
                        "hard_stop_distance",
                        getattr(self.cfg.safety, "hard_stop_distance", "None"),
                    ),
                ),
                side_points_count=scan_result.get("valid_side_count", 0),
                side_avoid_mode=planner_debug.get(
                    "side_avoid_mode",
                    scan_result.get("side_avoid_mode", "none"),
                ),
                side_stop_mode=planner_debug.get(
                    "side_stop_mode",
                    scan_result.get("side_stop_mode", "none"),
                ),
                side_obstacle_side=planner_debug.get(
                    "side_obstacle_side",
                    scan_result.get("side_obstacle_side", "none"),
                ),
                side_soft_distance=planner_debug.get(
                    "side_soft_distance",
                    scan_result.get(
                        "side_soft_distance",
                        getattr(self.cfg.safety, "side_soft_distance", 0.32),
                    ),
                ),
                side_hard_distance=planner_debug.get(
                    "side_hard_distance",
                    scan_result.get(
                        "side_hard_distance",
                        getattr(self.cfg.safety, "side_hard_distance", 0.18),
                    ),
                ),
                side_release_distance=planner_debug.get(
                    "side_release_distance",
                    scan_result.get(
                        "side_release_distance",
                        getattr(self.cfg.safety, "side_release_distance", 0.36),
                    ),
                ),
                omega_before_side_avoid=planner_debug.get(
                    "omega_before_side_avoid",
                    scan_result.get("omega_before_side_avoid", "None"),
                ),
                omega_after_side_avoid=planner_debug.get(
                    "omega_after_side_avoid",
                    scan_result.get("omega_after_side_avoid", "None"),
                ),
                side_avoid_applied=planner_debug.get(
                    "side_avoid_applied",
                    scan_result.get("side_avoid_applied", False),
                ),
                side_hard_stop_applied=planner_debug.get(
                    "side_hard_stop_applied",
                    scan_result.get("side_hard_stop_applied", False),
                ),
                soft_avoid_v_before=planner_debug.get("soft_avoid_v_before", "None"),
                soft_avoid_v_after=planner_debug.get("soft_avoid_v_after", "None"),
                soft_avoid_reason=planner_debug.get("soft_avoid_reason", "none"),
                front_slow_min_scale=planner_debug.get(
                    "front_slow_min_scale",
                    scan_result.get(
                        "front_slow_min_scale",
                        getattr(self.cfg.safety, "front_slow_min_scale", "None"),
                    ),
                ),
                side_soft_min_v=planner_debug.get(
                    "side_soft_min_v",
                    getattr(self.cfg.safety, "side_soft_min_v", "None"),
                ),
                scan_frame=self.last_scan_frame_id,
                base_frame=self.base_frame_id,
                scan_offset_deg=self.latest_scan_effective_offset_deg,
                scan_tf_ok=self.latest_scan_tf_ok,
                goal_distance=goal_distance_text,
                goal_bearing=goal_bearing_text,
                yaw_error_deg=yaw_error_text,
                tracking_mode=planner_debug.get("tracking_mode", "none"),
                tracking_v_scale=float(planner_debug.get("tracking_v_scale", 1.0)),
                omega_before_tracking=planner_debug.get("omega_before_tracking", "None"),
                omega_after_tracking=planner_debug.get("omega_after_tracking", "None"),
                tracking_omega_before=planner_debug.get("tracking_omega_before", "None"),
                tracking_omega_after=planner_debug.get("tracking_omega_after", "None"),
                goal_tracking_intervened=planner_debug.get(
                    "goal_tracking_intervened",
                    False,
                ),
                goal_tracking_reason=planner_debug.get("goal_tracking_reason", "none"),
                smoothing_reset_reason=planner_debug.get("smoothing_reset_reason", "none"),
                heading_gate_active=planner_debug.get("heading_gate_active", False),
                obstacle_mode=obstacle_mode,
                segments=obstacle_segment_count,
                use_anisotropic_sampling=getattr(
                    self.cfg.mppi,
                    "use_anisotropic_sampling",
                    False,
                ),
                sdf_influence_distance=getattr(
                    self.cfg.mppi,
                    "sdf_influence_distance",
                    "None",
                ),
                scan_obstacle_mode=getattr(
                    self.cfg.mppi,
                    "scan_obstacle_mode",
                    "raw",
                ),
                edges=obstacle_edge_count,
                line_surfaces=obstacle_line_count,
                circle_obstacles=obstacle_circle_count,
                edge_corners=obstacle_corner_count,
                irregular=obstacle_irregular_count,
                dynamic_obstacles=self.latest_dynamic_obstacle_count,
                dynamic_obstacles_current_scan=self.latest_dynamic_obstacle_current_scan_count,
                planner_obstacles=planner_obstacle_text,
                geometric_obstacle_count_raw=geometric_obstacle_count_raw,
                geometric_obstacle_count_used=geometric_obstacle_count_used,
                compressed_line_obstacles=compressed_line_obstacles,
                line_compression_mode=line_compression_mode,
                front_corridor_obstacle_count=planner_debug.get(
                    "front_corridor_obstacle_count",
                    0,
                ),
                front_corridor_min_range=front_corridor_min_range_text,
                obstacle_turn_mode=planner_debug.get("obstacle_turn_mode", False),
                preemptive_turn_candidate=planner_debug.get(
                    "preemptive_turn_candidate",
                    False,
                ),
                obstacle_context_gate_reason=planner_debug.get(
                    "obstacle_context_gate_reason",
                    "none",
                ),
                front_turn_distance=planner_debug.get(
                    "front_turn_distance",
                    getattr(self.cfg.safety, "front_turn_distance", 0.85),
                ),
                front_turn_release_distance=planner_debug.get(
                    "front_turn_release_distance",
                    getattr(self.cfg.safety, "front_turn_release_distance", 0.95),
                ),
                obstacle_turn_entry_reason=planner_debug.get(
                    "obstacle_turn_entry_reason",
                    "none",
                ),
                obstacle_turn_release_reason=planner_debug.get(
                    "obstacle_turn_release_reason",
                    "none",
                ),
                obstacle_turn_yaw_damped=planner_debug.get(
                    "obstacle_turn_yaw_damped",
                    False,
                ),
                obstacle_turn_yaw_released=planner_debug.get(
                    "obstacle_turn_yaw_released",
                    False,
                ),
                obstacle_turn_direction=planner_debug.get(
                    "obstacle_turn_direction",
                    "none",
                ),
                obstacle_turn_omega=planner_debug.get("obstacle_turn_omega", 0.0),
                omega_before_obstacle_turn=planner_debug.get(
                    "omega_before_obstacle_turn",
                    "None",
                ),
                omega_after_obstacle_turn=planner_debug.get(
                    "omega_after_obstacle_turn",
                    "None",
                ),
                omega_source=planner_debug.get("omega_source", "mppi"),
                mppi_omega_preserved=planner_debug.get("mppi_omega_preserved", True),
                mppi_omega_overridden=planner_debug.get("mppi_omega_overridden", False),
                omega_override_reason=planner_debug.get("omega_override_reason", "none"),
                proposed_omega=planner_debug.get("proposed_omega", "None"),
                final_omega_before_smoothing=planner_debug.get(
                    "final_omega_before_smoothing",
                    "None",
                ),
                final_omega_after_smoothing=planner_debug.get(
                    "final_omega_after_smoothing",
                    "None",
                ),
                smoothing_sign_flip_blocked=planner_debug.get(
                    "smoothing_sign_flip_blocked",
                    False,
                ),
                smoothing_omega_before_block=planner_debug.get(
                    "smoothing_omega_before_block",
                    "None",
                ),
                smoothing_omega_after_block=planner_debug.get(
                    "smoothing_omega_after_block",
                    "None",
                ),
                smoothing_block_reason=planner_debug.get(
                    "smoothing_block_reason",
                    "none",
                ),
                mppi_turn_direction=planner_debug.get("mppi_turn_direction", "none"),
                override_risk_score=planner_debug.get("override_risk_score", 0.0),
                turn_v_scale=planner_debug.get("turn_v_scale", 1.0),
                avoidance_direction_locked=planner_debug.get(
                    "avoidance_direction_locked",
                    False,
                ),
                avoidance_release_reason=planner_debug.get(
                    "avoidance_release_reason",
                    "none",
                ),
                left_corridor_score=planner_debug.get("left_corridor_score", 0.0),
                right_corridor_score=planner_debug.get("right_corridor_score", 0.0),
                corridor_selected_direction=planner_debug.get(
                    "corridor_selected_direction",
                    "none",
                ),
                corridor_switch_allowed=planner_debug.get(
                    "corridor_switch_allowed",
                    False,
                ),
                corridor_switch_reason=planner_debug.get(
                    "corridor_switch_reason",
                    "none",
                ),
                corridor_switch_margin=planner_debug.get("corridor_switch_margin", 0.0),
                corridor_switch_streak=planner_debug.get("corridor_switch_streak", 0),
                switch_blocked_by_lock=planner_debug.get("switch_blocked_by_lock", False),
                switch_block_reason=planner_debug.get("switch_block_reason", "none"),
                selected_direction_age_sec=planner_debug.get(
                    "selected_direction_age_sec",
                    0.0,
                ),
                arbitration_mode=planner_debug.get("arbitration_mode", "mppi"),
                hard_override_allowed=planner_debug.get("hard_override_allowed", False),
                hard_override_gate_reason=planner_debug.get(
                    "hard_override_gate_reason",
                    "none",
                ),
                mppi_direction_conflict=planner_debug.get(
                    "mppi_direction_conflict",
                    False,
                ),
                mppi_conflict_frames=planner_debug.get("mppi_conflict_frames", 0),
                min_front_range_trend=planner_debug.get(
                    "min_front_range_trend",
                    getattr(self, "min_front_range_trend", "unknown"),
                ),
                goal_distance_trend=planner_debug.get(
                    "goal_distance_trend",
                    getattr(self, "goal_distance_trend", "unknown"),
                ),
                avoidance_state=planner_debug.get("avoidance_state", "CLEAR"),
                avoidance_state_reason=planner_debug.get(
                    "avoidance_state_reason",
                    "none",
                ),
                avoidance_state_age_sec=planner_debug.get(
                    "avoidance_state_age_sec",
                    0.0,
                ),
                avoidance_release_candidate=planner_debug.get(
                    "avoidance_release_candidate",
                    False,
                ),
                avoidance_release_block_reason=planner_debug.get(
                    "avoidance_release_block_reason",
                    "none",
                ),
                clear_streak=planner_debug.get("clear_streak", 0),
                progress_streak=planner_debug.get("progress_streak", 0),
                stuck_streak=planner_debug.get("stuck_streak", 0),
                creep_escape_active=planner_debug.get("creep_escape_active", False),
                creep_v_selected=planner_debug.get("creep_v_selected", "None"),
                creep_omega_limit=planner_debug.get("creep_omega_limit", "None"),
                creep_escape_reason=planner_debug.get("creep_escape_reason", "none"),
                creep_progress_ok=planner_debug.get("creep_progress_ok", False),
                creep_front_range_delta=planner_debug.get(
                    "creep_front_range_delta",
                    "None",
                ),
                creep_goal_distance_delta=planner_debug.get(
                    "creep_goal_distance_delta",
                    "None",
                ),
                front_stop_release_distance=planner_debug.get(
                    "front_stop_release_distance",
                    getattr(self.cfg.safety, "front_stop_release_distance", 0.36),
                ),
                recovery_direction=planner_debug.get("recovery_direction", "none"),
                hard_stop_recovery_age_sec=planner_debug.get(
                    "hard_stop_recovery_age_sec",
                    0.0,
                ),
                hard_stop_recovery_direction=planner_debug.get(
                    "hard_stop_recovery_direction",
                    planner_debug.get("recovery_direction", "none"),
                ),
                hard_stop_recovery_switch_reason=planner_debug.get(
                    "hard_stop_recovery_switch_reason",
                    "none",
                ),
                hard_stop_release_candidate=planner_debug.get(
                    "hard_stop_release_candidate",
                    False,
                ),
                hard_stop_exit_to_creep=planner_debug.get("hard_stop_exit_to_creep", False),
                left_guard_count=planner_debug.get("left_guard_count", 0),
                right_guard_count=planner_debug.get("right_guard_count", 0),
                smoothing_context=planner_debug.get("smoothing_context", "clear"),
                smoothing_alpha_effective=planner_debug.get(
                    "smoothing_alpha_effective",
                    "None",
                ),
                smoothing_magnitude_clamped=planner_debug.get(
                    "smoothing_magnitude_clamped",
                    False,
                ),
                smoother_reset_on_direction_switch=planner_debug.get(
                    "smoother_reset_on_direction_switch",
                    False,
                ),
                goal_reacquire_active=planner_debug.get("goal_reacquire_active", False),
                anti_goal_drive_blocked=planner_debug.get("anti_goal_drive_blocked", False),
                heading_gate_reason=planner_debug.get("heading_gate_reason", "none"),
                goal_tracking_suppression_release_reason=planner_debug.get(
                    "goal_tracking_suppression_release_reason",
                    "none",
                ),
                degraded_level=planner_debug.get("degraded_level", "normal"),
                degraded_enter_reason=planner_debug.get("degraded_enter_reason", "none"),
                degraded_exit_reason=planner_debug.get("degraded_exit_reason", "none"),
                core_overrun_streak=planner_debug.get("core_overrun_streak", 0),
                core_recover_streak=planner_debug.get("core_recover_streak", 0),
                planner_skipped_no_core_update=planner_debug.get(
                    "planner_skipped_no_core_update",
                    False,
                ),
                line_fit_candidates=line_fit_candidates,
                line_fit_accepted=line_fit_accepted,
                line_fit_rejected_reason=line_fit_rejected_reason,
                line_surface_streak=line_surface_streak,
                representative_circle_count=representative_circle_count,
                obstacle_budget_mode=obstacle_budget_mode,
                obstacle_yaw_deadband_escape_applied=planner_debug.get(
                    "obstacle_yaw_deadband_escape_applied",
                    False,
                ),
                avoidance_side=self.avoidance_side,
                avoidance_override_applied=planner_debug.get("avoidance_override_applied", False),
                avoidance_hold_remaining=self.avoidance_hold_remaining(),
                raw_final_control=raw_final_text,
                smoothed_control=final_text,
                final_control=final_text,
                publish_enabled=self.enable_publish,
                note=note,
            )
        )

    def run_once(self):
        loop_start = time.time()
        self.current_loop_start_time = loop_start
        if self.previous_loop_start_time is None:
            self.current_loop_dt_ms = None
        else:
            self.current_loop_dt_ms = (
                loop_start - self.previous_loop_start_time
            ) * 1000.0
        self.previous_loop_start_time = loop_start

        if self.current_state_exp is None or self.is_odom_stale():
            self.clear_dynamic_obstacles()
            scan_result = self.latest_scan_result
            if scan_result is None or self.is_scan_stale():
                scan_result = make_no_scan_result()
            if self.current_state_exp is None:
                note = "waiting_for_odom_no_motion"
                reason = "no_odom"
            else:
                note = "odom_stale_no_motion"
                reason = "odom_stale"
            self.publish_zero_twist(reason=reason)
            self.print_cycle_status(
                state=self.current_state_exp,
                goal=self.goal_xy,
                proposed_control=None,
                scan_result=scan_result,
                final_control=(0.0, 0.0),
                note=note,
                planner_debug={
                    "planner_type": "not_called",
                    "front_stop_mode": scan_result.get("front_stop_mode", reason),
                    "planner_creep_mode": False,
                    "planner_called_under_front_block": False,
                    "planner_skip_reason": reason,
                    "planner_compute_ms": 0.0,
                },
            )
            return

        state = self.current_state_exp
        goal = self.goal_xy
        heading_debug = self.compute_heading_debug(state, goal)
        goal_distance = heading_debug["goal_distance"]
        self.publish_goal_bearing_debug(state, goal)

        scan_unavailable = self.latest_scan_result is None or self.is_scan_stale()
        if scan_unavailable:
            scan_result = make_no_scan_result()
        else:
            scan_result = self.latest_scan_result

        self.update_progress_trends(scan_result, heading_debug)
        avoidance_debug = self.update_avoidance_state(
            scan_result,
            heading_debug,
            planner_obstacles=self.latest_dynamic_obstacle_count,
            dynamic_obstacles_current_scan=self.latest_dynamic_obstacle_current_scan_count,
        )
        stop_reason = scan_result.get("reason", "")
        min_front_for_recovery = scan_result.get("min_front_range")
        front_stop_release_distance = float(
            getattr(self.cfg.safety, "front_stop_release_distance", 0.36)
        )
        hard_zero_reasons = (
            "no_scan_emergency_stop",
            "no_valid_front_scan",
            "scan_tf_failed",
            "near_body_stop",
        )
        if scan_unavailable or stop_reason in hard_zero_reasons:
            self.front_stop_recovery_active = False
            self.front_stop_recovery_direction = "none"
            self.clear_dynamic_obstacles()
            reason = scan_result.get("reason", "scan_guard_stop")
            self.publish_zero_twist(reason=reason)
            self.publish_final_cmd_debug((0.0, 0.0), reason)
            self.print_cycle_status(
                state=state,
                goal=goal,
                proposed_control=None,
                scan_result=scan_result,
                final_control=(0.0, 0.0),
                note=reason,
                planner_debug=dict(
                    avoidance_debug,
                    **{
                    "planner_type": "not_called",
                    "goal_distance": goal_distance,
                    "goal_bearing": heading_debug["goal_bearing"],
                    "yaw_error_deg": heading_debug["yaw_error_deg"],
                    "heading_gate_active": False,
                    "raw_final_control": (0.0, 0.0),
                    "tracking_mode": "hard_stop",
                    "tracking_v_scale": 0.0,
                    "front_stop_mode": scan_result.get("front_stop_mode", reason),
                    "planner_creep_mode": False,
                    "planner_called_under_front_block": False,
                    "planner_skip_reason": reason,
                    "planner_compute_ms": 0.0,
                    "omega_before_tracking": 0.0,
                    "omega_after_tracking": 0.0,
                    "smoothing_reset_reason": "emergency_stop",
                    }
                ),
            )
            return

        front_hard_stop_reasons = ("hard_stop", "front_obstacle_stop", "near_body_hard_stop")
        hard_stop_release_margin = float(
            getattr(self.cfg.safety, "hard_stop_release_margin", 0.04)
        )
        hard_stop_release_distance = float(
            getattr(self.cfg.safety, "hard_stop_distance", 0.18)
        ) + max(0.0, hard_stop_release_margin)
        if stop_reason in front_hard_stop_reasons:
            self.front_stop_recovery_active = True
        elif self.front_stop_recovery_active:
            if (
                min_front_for_recovery is not None
                and float(min_front_for_recovery) > hard_stop_release_distance
            ):
                self.front_stop_recovery_active = False
                self.front_stop_recovery_direction = "none"
                self.set_avoidance_state("CREEP_ESCAPE", "hard_stop_exit_to_creep")
                avoidance_debug = self.current_avoidance_debug(
                    release_candidate=False,
                    release_block_reason="hard_stop_exit_to_creep",
                    hard_stop_release_candidate=True,
                )
            else:
                stop_reason = "hard_stop"

        if stop_reason in front_hard_stop_reasons:
            self.clear_planner_obstacles_only()
            recovery_control, recovery_active, recovery_info = self.front_stop_rotate_recovery(
                heading_debug,
                scan_result,
            )
            recovery_control = self.final_safety_clamp(recovery_control)
            recovery_reset_omega = recovery_control[1] if recovery_active else 0.0
            self.reset_omega_smoother(recovery_reset_omega)
            final_control, smooth_debug = self.smooth_control_for_publish(
                recovery_control,
                force_zero=not recovery_active,
                context="hard_stop_recovery",
            )
            final_control = self.final_safety_clamp(final_control)
            self.maybe_publish_control(control_to_twist_dict(final_control))
            note = "front_stop_rotate_recovery" if recovery_active else "hard_stop"
            self.publish_final_cmd_debug(final_control, note)
            planner_debug = {
                "planner_type": "not_called",
                "goal_distance": goal_distance,
                "goal_bearing": heading_debug["goal_bearing"],
                "yaw_error_deg": heading_debug["yaw_error_deg"],
                "heading_gate_active": False,
                "raw_final_control": recovery_control,
                "tracking_mode": "front_stop_recovery",
                "tracking_v_scale": 0.0,
                "front_stop_mode": "hard_stop",
                "planner_creep_mode": False,
                "planner_called_under_front_block": False,
                "planner_skip_reason": stop_reason,
                "planner_compute_ms": 0.0,
                "omega_before_tracking": 0.0,
                "omega_after_tracking": recovery_control[1],
                "smoothing_reset_reason": "front_stop_recovery",
                "avoidance_override_applied": False,
                "control_smooth": smooth_debug,
                "front_stop_release_distance": recovery_info.get(
                    "front_stop_release_distance",
                    front_stop_release_distance,
                ),
                "recovery_direction": recovery_info.get("recovery_direction", "none"),
                "left_guard_count": recovery_info.get("left_guard_count", 0),
                "right_guard_count": recovery_info.get("right_guard_count", 0),
                "obstacle_yaw_deadband_escape_applied": False,
            }
            planner_debug.update(avoidance_debug)
            planner_debug.update(recovery_info)
            self.print_cycle_status(
                state=state,
                goal=goal,
                proposed_control=None,
                scan_result=scan_result,
                final_control=final_control,
                note=note,
                planner_debug=planner_debug,
            )
            return

        if stop_reason == "side_hard_stop":
            self.front_stop_recovery_active = False
            self.front_stop_recovery_direction = "none"
            self.clear_dynamic_obstacles()
            recovery_control, recovery_active, recovery_info = self.side_hard_stop_recovery(
                scan_result,
            )
            recovery_control = self.final_safety_clamp(recovery_control)
            recovery_reset_omega = recovery_control[1] if recovery_active else 0.0
            self.reset_omega_smoother(recovery_reset_omega)
            final_control, smooth_debug = self.smooth_control_for_publish(
                recovery_control,
                force_zero=not recovery_active,
                context="hard_stop_recovery",
            )
            final_control = self.final_safety_clamp(final_control)
            self.maybe_publish_control(control_to_twist_dict(final_control))
            note = "side_hard_recovery" if recovery_active else "side_hard_stop"
            self.publish_final_cmd_debug(final_control, note)
            planner_debug = {
                "planner_type": "not_called",
                "goal_distance": goal_distance,
                "goal_bearing": heading_debug["goal_bearing"],
                "yaw_error_deg": heading_debug["yaw_error_deg"],
                "heading_gate_active": False,
                "raw_final_control": recovery_control,
                "tracking_mode": "side_hard_recovery" if recovery_active else "side_hard_stop",
                "tracking_v_scale": 0.0,
                "front_stop_mode": scan_result.get("front_stop_mode", "side_hard_stop"),
                "planner_creep_mode": False,
                "planner_called_under_front_block": False,
                "planner_skip_reason": "side_hard_stop",
                "planner_compute_ms": 0.0,
                "omega_before_tracking": 0.0,
                "omega_after_tracking": recovery_control[1],
                "smoothing_reset_reason": "side_hard_stop",
                "avoidance_override_applied": False,
                "control_smooth": smooth_debug,
                "obstacle_yaw_deadband_escape_applied": False,
                "front_stop_release_distance": front_stop_release_distance,
            }
            planner_debug.update(avoidance_debug)
            planner_debug.update(recovery_info)
            self.print_cycle_status(
                state=state,
                goal=goal,
                proposed_control=None,
                scan_result=scan_result,
                final_control=final_control,
                note=note,
                planner_debug=planner_debug,
            )
            return

        if self.goal_reached_latched or goal_distance <= float(self.cfg.goal.tolerance):
            self.goal_reached_latched = True
            self.front_stop_recovery_active = False
            self.front_stop_recovery_direction = "none"
            self.clear_dynamic_obstacles()
            self.publish_zero_twist(reason="goal_reached_latched")
            self.publish_final_cmd_debug((0.0, 0.0), "goal_reached_latched")
            self.print_cycle_status(
                state=state,
                goal=goal,
                proposed_control=(0.0, 0.0),
                scan_result=scan_result,
                final_control=(0.0, 0.0),
                note="goal_reached_latched",
                planner_debug={
                    "avoidance_state": avoidance_debug.get("avoidance_state", "CLEAR"),
                    "avoidance_state_reason": avoidance_debug.get("avoidance_state_reason", "none"),
                    "avoidance_state_age_sec": avoidance_debug.get("avoidance_state_age_sec", 0.0),
                    "avoidance_release_candidate": avoidance_debug.get("avoidance_release_candidate", False),
                    "avoidance_release_block_reason": avoidance_debug.get("avoidance_release_block_reason", "none"),
                    "clear_streak": avoidance_debug.get("clear_streak", 0),
                    "progress_streak": avoidance_debug.get("progress_streak", 0),
                    "stuck_streak": avoidance_debug.get("stuck_streak", 0),
                    "planner_type": "not_called",
                    "goal_distance": goal_distance,
                    "goal_bearing": heading_debug["goal_bearing"],
                    "yaw_error_deg": heading_debug["yaw_error_deg"],
                    "heading_gate_active": False,
                    "raw_final_control": (0.0, 0.0),
                    "tracking_mode": "goal_reached_latched",
                    "tracking_v_scale": 0.0,
                    "front_stop_mode": scan_result.get(
                        "front_stop_mode",
                        scan_result.get("reason", "none"),
                    ),
                    "planner_creep_mode": False,
                    "planner_called_under_front_block": False,
                    "planner_skip_reason": "goal_reached_latched",
                    "planner_compute_ms": 0.0,
                    "omega_before_tracking": 0.0,
                    "omega_after_tracking": 0.0,
                    "smoothing_reset_reason": "goal_reached_latched",
                },
            )
            return

        try:
            planner_call_start = time.time()
            if self.planner_mode == "fake":
                self.clear_dynamic_obstacles()
                proposed_control = fake_planner_control(state, goal)
                planner_debug = {
                    "planner_type": "fake",
                    "proposed_control": proposed_control,
                    "planner_compute_ms": 0.0,
                }
            else:
                self.apply_mppi_runtime_profile()
                self.refresh_mppi_dynamic_obstacles(
                    state=state,
                    scan_unavailable=scan_unavailable,
                )
                avoidance_debug = self.update_avoidance_state(
                    scan_result,
                    heading_debug,
                    planner_obstacles=self.latest_dynamic_obstacle_count,
                    dynamic_obstacles_current_scan=self.latest_dynamic_obstacle_current_scan_count,
                )
                proposed_control, planner_debug = self.mppi_bridge.compute_control(
                    current_state_exp=state,
                    goal_xy=goal,
                )
            planner_elapsed_ms = (time.time() - planner_call_start) * 1000.0
            planner_core_compute_ms = planner_debug.get("planner_compute_ms")
            if planner_core_compute_ms is None and planner_debug.get("plan_time_sec") is not None:
                planner_core_compute_ms = float(planner_debug.get("plan_time_sec")) * 1000.0
            if planner_core_compute_ms is None:
                planner_core_compute_ms = planner_elapsed_ms
            planner_debug["planner_core_compute_ms"] = float(planner_core_compute_ms)
            planner_debug["planner_total_ms"] = planner_elapsed_ms
            planner_debug["planner_compute_ms"] = max(
                float(planner_core_compute_ms),
                planner_elapsed_ms,
            )
        except Exception as exc:
            self.rospy.logerr(
                "[mppi_ros_adapter_skeleton] planner failed; publishing zero Twist. "
                "Original error: {}".format(exc)
            )
            self.publish_zero_twist(reason="planner_error")
            self.print_cycle_status(
                state=state,
                goal=goal,
                proposed_control=None,
                scan_result=scan_result,
                final_control=(0.0, 0.0),
                note="planner_error",
                planner_debug={
                    "planner_type": self.planner_mode,
                    "front_stop_mode": scan_result.get(
                        "front_stop_mode",
                        scan_result.get("reason", "none"),
                    ),
                    "planner_creep_mode": False,
                    "planner_called_under_front_block": False,
                    "planner_skip_reason": "planner_error",
                    "planner_compute_ms": 0.0,
                },
            )
            return

        if bool(self.cfg.safety.enable_scan_guard):
            emergency_reason = scan_result["reason"]
            slow_scale = float(scan_result["slow_scale"])
        else:
            emergency_reason = "scan_guard_disabled"
            slow_scale = 1.0

        safe_control, _, debug = prepare_safe_command(
            proposed_control=proposed_control,
            v_max=self.cfg.limits.v_max,
            w_max=self.cfg.limits.w_max,
            allow_backward=False,
            slow_scale=slow_scale,
            emergency_stop=False,
            emergency_reason=emergency_reason,
        )

        intervention_reason = emergency_reason
        planner_called_under_front_block = (
            scan_result.get("front_stop_mode") == "front_soft_block"
            or scan_result.get("reason") == "front_soft_block"
        )
        front_creep_control, front_creep_debug = self.apply_front_soft_block_creep(
            safe_control,
            scan_result,
        )
        obstacle_turn_preview = self.obstacle_turn_activation_info(
            scan_result,
            self.latest_dynamic_obstacle_count,
        )
        obstacle_turn_control, obstacle_turn_debug = self.apply_preemptive_obstacle_turn(
            front_creep_control,
            scan_result,
            self.latest_dynamic_obstacle_count,
            heading_debug=heading_debug,
            activation_info=obstacle_turn_preview,
        )
        side_avoid_control, side_avoid_debug = self.apply_side_soft_avoid(
            obstacle_turn_control,
            scan_result,
            obstacle_turn_debug=obstacle_turn_debug,
        )
        soft_floor_control, soft_avoid_debug = self.apply_soft_avoid_velocity_floor(
            side_avoid_control,
            scan_result,
            side_avoid_debug=side_avoid_debug,
        )
        tracking_control, tracking_debug = self._apply_goal_tracking_override(
            state,
            goal,
            soft_floor_control,
            self.cfg,
            {
                "scan_result": scan_result,
                "dynamic_obstacles": self.latest_dynamic_obstacle_count,
                "dynamic_obstacles_current_scan": self.latest_dynamic_obstacle_current_scan_count,
                "planner_obstacles": self.latest_dynamic_obstacle_count,
                "obstacle_mode": (self.latest_dynamic_obstacle_debug or {}).get(
                    "mode",
                    getattr(self.cfg.mppi, "scan_obstacle_mode", "raw"),
                ),
                "proposed_omega": proposed_control[1],
                "avoidance_side": self.avoidance_side,
                "avoidance_state": avoidance_debug.get("avoidance_state", self.avoidance_state),
                "obstacle_turn_mode": obstacle_turn_debug.get("obstacle_turn_mode", False),
                "side_soft_avoid": side_avoid_debug.get("side_avoid_applied", False),
            },
        )
        smoothing_reset_reason = tracking_debug.get("smoothing_reset_reason", "none")
        smoother_reset_on_direction_switch = False
        if obstacle_turn_debug.get("obstacle_turn_mode"):
            front_turn_min_omega = float(
                getattr(self.cfg.safety, "front_turn_min_omega", 0.12)
            )
            if (
                obstacle_turn_debug.get("omega_source") == "hard_override"
                and (
                    abs(self.last_smoothed_omega) < front_turn_min_omega
                    or sign_of(self.last_smoothed_omega) != sign_of(obstacle_turn_control[1])
                    or abs(obstacle_turn_control[1]) > abs(self.last_smoothed_omega) + 0.03
                )
            ):
                smoothing_reset_reason = "preemptive_obstacle_turn"
                tracking_debug["smoothing_reset_reason"] = smoothing_reset_reason
        elif obstacle_turn_debug.get("obstacle_turn_release_reason", "none") != "none":
            if smoothing_reset_reason == "none":
                smoothing_reset_reason = "obstacle_turn_release"
                tracking_debug["smoothing_reset_reason"] = smoothing_reset_reason
        if obstacle_turn_debug.get("corridor_direction_switched", False):
            smoothing_reset_reason = "corridor_direction_switch"
            tracking_debug["smoothing_reset_reason"] = smoothing_reset_reason
            smoother_reset_on_direction_switch = True
        if side_avoid_debug.get("side_avoid_applied"):
            if (
                sign_of(self.last_smoothed_omega) != sign_of(side_avoid_control[1])
                or abs(side_avoid_control[1]) > abs(self.last_smoothed_omega) + 0.03
            ):
                smoothing_reset_reason = "side_soft_avoid"
                tracking_debug["smoothing_reset_reason"] = smoothing_reset_reason
        if smoothing_reset_reason != "none":
            reset_omega = tracking_control[1]
            if smoothing_reset_reason in (
                "yaw_deadband",
                "emergency_stop",
                "goal_reached_latched",
            ):
                reset_omega = 0.0
            self.reset_omega_smoother(reset_omega)
        if tracking_debug.get("tracking_mode") not in ("pass_through", "disabled"):
            intervention_reason = tracking_debug.get("tracking_mode", intervention_reason)
        if obstacle_turn_debug.get("obstacle_turn_mode"):
            intervention_reason = "preemptive_obstacle_turn"
        if side_avoid_debug.get("side_avoid_applied"):
            intervention_reason = "side_soft_avoid"

        smoothing_context = "clear"
        if avoidance_debug.get("avoidance_state") == "CREEP_ESCAPE":
            smoothing_context = "creep"
        elif avoidance_debug.get("avoidance_state") in ("APPROACH_SLOW", "GOAL_REACQUIRE"):
            smoothing_context = "obstacle"
        elif (
            obstacle_turn_debug.get("obstacle_turn_mode")
            or side_avoid_debug.get("side_avoid_applied")
        ):
            smoothing_context = "obstacle"

        final_control, smooth_debug = self.smooth_control_for_publish(
            tracking_control,
            force_zero=False,
            context=smoothing_context,
            goal_tracking_active=bool(
                tracking_debug.get("goal_tracking_intervened", False)
                or tracking_debug.get("tracking_mode")
                in (
                    "heading_align_soft",
                    "heading_slow_gate",
                    "goal_track_override",
                    "goal_track_blend",
                )
            ),
            yaw_error_deg=float(tracking_debug.get("yaw_error_deg", 0.0)),
        )
        smooth_debug["smoother_reset_on_direction_switch"] = smoother_reset_on_direction_switch
        final_control, smoothing_block_debug = self.block_smoothing_sign_flip_if_needed(
            tracking_control,
            final_control,
            scan_result,
            obstacle_turn_debug,
            side_avoid_debug,
            front_creep_debug,
            planner_debug,
        )
        final_control = self.final_safety_clamp(final_control)
        if tracking_debug.get("tracking_mode") == "yaw_deadband":
            omega_deadband_max = float(
                getattr(self.cfg.tracking, "omega_deadband_max", 0.12)
            )
            final_control = (
                final_control[0],
                clip_value(final_control[1], -omega_deadband_max, omega_deadband_max),
            )
        twist_dict = control_to_twist_dict(final_control)

        final_omega_source = obstacle_turn_debug.get("omega_source", "mppi")
        if side_avoid_debug.get("side_avoid_applied"):
            if abs(float(side_avoid_control[1]) - float(obstacle_turn_control[1])) > 1e-6:
                final_omega_source = "safety_clamped"
        if (
            tracking_debug.get("goal_tracking_intervened", False)
            and not obstacle_turn_debug.get("obstacle_turn_mode", False)
            and not side_avoid_debug.get("side_avoid_applied", False)
        ):
            final_omega_source = "goal_tracking"

        planner_debug["raw_final_control"] = tracking_control
        planner_debug["smoothed_control"] = final_control
        planner_debug["control_intervention_reason"] = intervention_reason
        planner_debug["control_smooth"] = smooth_debug
        planner_debug.update(avoidance_debug)
        planner_debug["avoidance_side"] = self.avoidance_side
        planner_debug["avoidance_hold_remaining"] = self.avoidance_hold_remaining()
        planner_debug["avoidance_override_applied"] = bool(
            obstacle_turn_debug.get("obstacle_turn_mode", False)
            or side_avoid_debug.get("side_avoid_applied", False)
        )
        planner_debug["front_stop_mode"] = scan_result.get(
            "front_stop_mode",
            scan_result.get("reason", "none"),
        )
        planner_debug["planner_called_under_front_block"] = planner_called_under_front_block
        planner_debug["planner_skip_reason"] = "none"
        planner_debug["final_omega_before_smoothing"] = tracking_control[1]
        planner_debug["final_omega_after_smoothing"] = final_control[1]
        planner_debug.update(obstacle_turn_debug)
        planner_debug.update(side_avoid_debug)
        planner_debug.update(front_creep_debug)
        planner_debug.update(soft_avoid_debug)
        planner_debug.update(tracking_debug)
        planner_debug.update(smoothing_block_debug)
        planner_debug["smoothing_context"] = smooth_debug.get(
            "smoothing_context",
            smoothing_block_debug.get("smoothing_context", "clear"),
        )
        planner_debug["smoothing_alpha_effective"] = smooth_debug.get(
            "smoothing_alpha_effective",
            planner_debug.get("smoothing_alpha_effective"),
        )
        planner_debug["smoothing_magnitude_clamped"] = smooth_debug.get(
            "smoothing_magnitude_clamped",
            False,
        )
        planner_debug["smoother_reset_on_direction_switch"] = smooth_debug.get(
            "smoother_reset_on_direction_switch",
            False,
        )
        planner_debug["omega_source"] = final_omega_source
        planner_debug["proposed_omega"] = proposed_control[1]
        if (
            tracking_debug.get("goal_tracking_intervened", False)
            and abs(float(tracking_control[1]) - float(soft_floor_control[1])) > 1e-6
        ):
            planner_debug["mppi_omega_preserved"] = False
            if (
                final_omega_source == "goal_tracking"
                and sign_of(proposed_control[1]) != 0
                and sign_of(tracking_control[1]) != 0
            ):
                if sign_of(proposed_control[1]) != sign_of(tracking_control[1]):
                    planner_debug["mppi_omega_overridden"] = True
                    planner_debug["omega_override_reason"] = "goal_tracking_reversed_mppi_omega"
                elif planner_debug.get("omega_override_reason", "none") == "none":
                    planner_debug["omega_override_reason"] = tracking_debug.get(
                        "goal_tracking_reason",
                        "goal_tracking_same_sign_limit",
                    )
        planner_debug["final_omega_before_smoothing"] = tracking_control[1]
        planner_debug["final_omega_after_smoothing"] = final_control[1]
        planner_debug["heading_gate_active"] = tracking_debug.get(
            "tracking_mode"
        ) in ("heading_align_soft", "heading_slow_gate")

        self.maybe_publish_control(twist_dict)
        self.publish_final_cmd_debug(final_control, intervention_reason)

        self.print_cycle_status(
            state=state,
            goal=goal,
            proposed_control=proposed_control,
            scan_result=scan_result,
            final_control=final_control,
            note="dry_run" if not self.enable_publish else "published",
            planner_debug=planner_debug,
        )

    def run(self):
        rate = self.rospy.Rate(self.rate_hz)

        try:
            while not self.rospy.is_shutdown():
                self.run_once()
                rate.sleep()
        except KeyboardInterrupt:
            self.rospy.logwarn("[mppi_ros_adapter_skeleton] Interrupted by user.")
        finally:
            self.publish_zero_twist(reason="shutdown")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Review skeleton for MPPI ROS hardware adapter."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to lab_runtime.yaml.",
    )
    parser.add_argument(
        "--front-angle-offset-deg",
        type=float,
        default=None,
        help="Laser front direction offset in degrees. Default: YAML safety.scan_angle_offset_deg.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Adapter loop rate in Hz. Default: 5.",
    )
    parser.add_argument(
        "--planner-mode",
        choices=["fake", "mppi"],
        default="fake",
        help="Planner backend. Default: fake.",
    )
    parser.add_argument(
        "--enable-publish",
        action="store_true",
        help="Override cfg.safety.enable_publish and publish /cmd_vel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Override cfg.safety.enable_publish and never publish /cmd_vel.",
    )

    args = parser.parse_args()
    if args.enable_publish and args.dry_run:
        parser.error("--enable-publish and --dry-run cannot be used together.")

    return args


def main():
    args = parse_args()

    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
    except ImportError as exc:
        print_ros_import_error(exc)
        return 1

    try:
        from scenario_config import load_lab_runtime_config
        cfg = load_lab_runtime_config(args.config)
    except Exception as exc:
        print_config_error(exc)
        return 1

    if args.enable_publish:
        enable_publish_override = True
    elif args.dry_run:
        enable_publish_override = False
    else:
        enable_publish_override = None

    rospy.init_node("mppi_ros_adapter_skeleton", anonymous=True)

    try:
        adapter = MppiRosAdapterSkeleton(
            rospy=rospy,
            odom_type=Odometry,
            scan_type=LaserScan,
            goal_type=PoseStamped,
            twist_type=Twist,
            cfg=cfg,
            front_angle_offset_deg=args.front_angle_offset_deg,
            rate_hz=args.rate,
            enable_publish_override=enable_publish_override,
            planner_mode=args.planner_mode,
        )
    except Exception as exc:
        if args.planner_mode == "mppi":
            print_planner_bridge_error(exc)
        else:
            print_config_error(exc)
        return 1

    adapter.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
