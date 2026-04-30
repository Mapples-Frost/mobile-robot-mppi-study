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

from control_adapter import prepare_safe_command
from frame_transform import ExperimentFrameTransform
from frame_transform import normalize_angle
from local_obstacle_layer import scan_to_experiment_obstacles
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
        "valid_front_count": 0,
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
        self.front_angle_offset_deg = float(front_angle_offset_deg)
        self.rate_hz = max(1.0, float(rate_hz))
        self.planner_mode = str(planner_mode).strip().lower()
        self.mppi_bridge = None

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
        self.odom_timeout_sec = 1.0
        self.scan_timeout_sec = 1.0
        self.goal_xy = (float(cfg.goal.x), float(cfg.goal.y))

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
            "planner_mode={}".format(
                cfg.ros.odom_topic,
                cfg.ros.scan_topic,
                cfg.ros.runtime_goal_topic,
                cfg.ros.cmd_vel_topic,
                self.planner_mode,
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

    def scan_callback(self, msg):
        self.latest_scan_msg_fields = {
            "ranges": list(msg.ranges),
            "angle_min": msg.angle_min,
            "angle_increment": msg.angle_increment,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
        }
        self.latest_scan_result = analyze_scan_front_sector(
            ranges=msg.ranges,
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
            range_min=msg.range_min,
            range_max=msg.range_max,
            front_stop_distance=self.cfg.safety.front_stop_distance,
            front_slow_distance=self.cfg.safety.front_slow_distance,
            front_angle_deg=self.cfg.safety.front_angle_deg,
            front_angle_offset_deg=self.front_angle_offset_deg,
        )
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

        if self.planner_mode != "mppi" or self.mppi_bridge is None:
            return

        if hasattr(self.mppi_bridge, "clear_obstacles"):
            self.mppi_bridge.clear_obstacles()
        else:
            self.mppi_bridge.set_obstacles([])

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

        try:
            dynamic_obstacles = scan_to_experiment_obstacles(
                ranges=self.latest_scan_msg_fields["ranges"],
                angle_min=self.latest_scan_msg_fields["angle_min"],
                angle_increment=self.latest_scan_msg_fields["angle_increment"],
                range_min=self.latest_scan_msg_fields["range_min"],
                range_max=self.latest_scan_msg_fields["range_max"],
                current_state_exp=state,
                max_radius=max_radius,
                min_radius=0.10,
                angle_offset_rad=math.radians(self.front_angle_offset_deg),
                downsample_step=downsample_step,
                obstacle_radius=self.cfg.safety.local_obstacle_radius,
            )
        except Exception:
            self.clear_dynamic_obstacles()
            raise

        # The config name is historical. In this first pass, enabling it feeds
        # scan circles into the existing MPPI obstacles input; local_obstacle_cost.py
        # is intentionally not wired into trajectory_cost here.
        self.mppi_bridge.set_obstacles(dynamic_obstacles)
        self.latest_local_obstacle_count = len(dynamic_obstacles)
        self.latest_dynamic_obstacle_count = len(dynamic_obstacles)

    def publish_zero_twist(self, reason=""):
        zero_twist = {
            "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

        if self.enable_publish and self.cmd_pub is not None:
            self.cmd_pub.publish(make_twist_from_dict(self.twist_type, zero_twist))

        if reason:
            self.rospy.logwarn(
                "[mppi_ros_adapter_skeleton] zero Twist requested: {}".format(reason)
            )

    def maybe_publish_control(self, twist_dict):
        if not self.enable_publish or self.cmd_pub is None:
            return

        self.cmd_pub.publish(make_twist_from_dict(self.twist_type, twist_dict))

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

        min_range = scan_result.get("min_front_range")
        if min_range is None:
            min_range_text = "None"
        else:
            min_range_text = "{:.3f}".format(min_range)

        if planner_debug is None:
            planner_debug = {}

        planner_type = planner_debug.get("planner_type", self.planner_mode)
        planner_obstacle_count = planner_debug.get("obstacle_count")
        if planner_obstacle_count is None:
            planner_obstacle_text = "None"
        else:
            planner_obstacle_text = str(planner_obstacle_count)

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

        self.rospy.loginfo(
            "[mppi_ros_adapter_skeleton] state={state} | "
            "goal=({goal_x:.3f}, {goal_y:.3f}) | planner={planner} | "
            "proposed_control={proposed_control} | "
            "scan_reason={reason} | emergency_stop={emergency_stop} | "
            "slow_scale={slow_scale:.3f} | min_front_range={min_range} | "
            "dynamic_obstacles={dynamic_obstacles} | "
            "planner_obstacles={planner_obstacles} | "
            "final_control={final_control} | publish_enabled={publish_enabled} | "
            "note={note}".format(
                state=state_text,
                goal_x=goal[0],
                goal_y=goal[1],
                planner=planner_text,
                proposed_control=proposed_text,
                reason=scan_result.get("reason"),
                emergency_stop=scan_result.get("emergency_stop"),
                slow_scale=float(scan_result.get("slow_scale", 0.0)),
                min_range=min_range_text,
                dynamic_obstacles=self.latest_dynamic_obstacle_count,
                planner_obstacles=planner_obstacle_text,
                final_control=final_text,
                publish_enabled=self.enable_publish,
                note=note,
            )
        )

    def run_once(self):
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
                planner_debug={"planner_type": "not_called"},
            )
            return

        state = self.current_state_exp
        goal = self.goal_xy
        goal_distance = math.hypot(goal[0] - state[0], goal[1] - state[1])

        if goal_distance <= float(self.cfg.goal.tolerance):
            if self.latest_scan_result is None or self.is_scan_stale():
                scan_result = make_no_scan_result()
                self.clear_dynamic_obstacles()
            else:
                scan_result = self.latest_scan_result
            self.publish_zero_twist(reason="goal_reached")
            self.print_cycle_status(
                state=state,
                goal=goal,
                proposed_control=(0.0, 0.0),
                scan_result=scan_result,
                final_control=(0.0, 0.0),
                note="goal_reached",
                planner_debug={"planner_type": "not_called"},
            )
            return

        scan_unavailable = self.latest_scan_result is None or self.is_scan_stale()
        if scan_unavailable:
            scan_result = make_no_scan_result()
        else:
            scan_result = self.latest_scan_result

        try:
            if self.planner_mode == "fake":
                self.clear_dynamic_obstacles()
                proposed_control = fake_planner_control(state, goal)
                planner_debug = {
                    "planner_type": "fake",
                    "proposed_control": proposed_control,
                }
            else:
                self.refresh_mppi_dynamic_obstacles(
                    state=state,
                    scan_unavailable=scan_unavailable,
                )
                proposed_control, planner_debug = self.mppi_bridge.compute_control(
                    current_state_exp=state,
                    goal_xy=goal,
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
                planner_debug={"planner_type": self.planner_mode},
            )
            return

        if scan_unavailable:
            emergency_stop = True
            emergency_reason = "no_scan"
            slow_scale = 0.0
        elif bool(self.cfg.safety.enable_scan_guard):
            emergency_stop = bool(scan_result["emergency_stop"])
            emergency_reason = scan_result["reason"]
            slow_scale = float(scan_result["slow_scale"])
        else:
            emergency_stop = False
            emergency_reason = "scan_guard_disabled"
            slow_scale = 1.0

        final_control, twist_dict, debug = prepare_safe_command(
            proposed_control=proposed_control,
            v_max=self.cfg.limits.v_max,
            w_max=self.cfg.limits.w_max,
            allow_backward=False,
            slow_scale=slow_scale,
            emergency_stop=emergency_stop,
            emergency_reason=emergency_reason,
        )

        self.maybe_publish_control(twist_dict)

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
        default=0.0,
        help="Laser front direction offset in degrees.",
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
