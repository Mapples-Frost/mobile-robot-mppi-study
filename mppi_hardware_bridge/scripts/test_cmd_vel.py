# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from control_adapter import prepare_safe_command


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


def make_twist_msg(twist_type, v, omega):
    msg = twist_type()
    msg.linear.x = float(v)
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = float(omega)
    return msg


def publish_zero_twist(rospy, publisher, twist_type, rate_hz, count=5):
    rate = rospy.Rate(max(float(rate_hz), 1.0))
    zero_msg = make_twist_msg(twist_type, 0.0, 0.0)

    for _ in range(int(count)):
        if rospy.is_shutdown():
            break
        publisher.publish(zero_msg)
        rate.sleep()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-speed /cmd_vel smoke test. This does not use MPPI."
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to lab_runtime.yaml.",
    )
    parser.add_argument(
        "--linear",
        type=float,
        default=0.03,
        help="Requested linear velocity in m/s. Default: 0.03.",
    )
    parser.add_argument(
        "--angular",
        type=float,
        default=0.0,
        help="Requested angular velocity in rad/s. Default: 0.0.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Publish duration in seconds. Default: 2.0.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Publish rate in Hz. Default: 10.",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Actually publish Twist messages. Without this flag it is a dry run.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    duration = max(0.0, float(args.duration))
    rate_hz = max(1.0, float(args.rate))

    try:
        import rospy
        from geometry_msgs.msg import Twist
    except ImportError as exc:
        print_ros_import_error(exc)
        return 1

    try:
        from scenario_config import load_lab_runtime_config
        cfg = load_lab_runtime_config(args.config)
    except Exception as exc:
        print_config_error(exc)
        return 1

    final_control, twist_dict, debug = prepare_safe_command(
        proposed_control=(args.linear, args.angular),
        v_max=cfg.limits.v_max,
        w_max=cfg.limits.w_max,
        allow_backward=False,
        slow_scale=1.0,
        emergency_stop=False,
        emergency_reason="",
    )
    final_v, final_omega = final_control

    print("")
    print("test_cmd_vel.py")
    print("----------------")
    print("topic: {}".format(cfg.ros.cmd_vel_topic))
    print("requested linear: {:.4f} m/s".format(float(args.linear)))
    print("requested angular: {:.4f} rad/s".format(float(args.angular)))
    print("clipped linear: {:.4f} m/s".format(final_v))
    print("clipped angular: {:.4f} rad/s".format(final_omega))
    print("duration: {:.3f} s".format(duration))
    print("rate: {:.3f} Hz".format(rate_hz))
    print("limits.v_max: {:.4f} m/s".format(float(cfg.limits.v_max)))
    print("limits.w_max: {:.4f} rad/s".format(float(cfg.limits.w_max)))
    print("")

    if not args.enable:
        print("DRY RUN: --enable was not provided.")
        print("No Twist message will be published.")
        print("If enabled, the script would publish:")
        print("  Twist.linear.x = {:.4f}".format(twist_dict["linear"]["x"]))
        print("  Twist.angular.z = {:.4f}".format(twist_dict["angular"]["z"]))
        print("")
        return 0

    rospy.init_node("test_cmd_vel", anonymous=True)
    publisher = rospy.Publisher(cfg.ros.cmd_vel_topic, Twist, queue_size=1)

    rospy.logwarn(
        "[test_cmd_vel] ENABLED. Publishing low-speed Twist on {}.".format(
            cfg.ros.cmd_vel_topic
        )
    )
    rospy.logwarn(
        "[test_cmd_vel] Before running this, make sure no other node controls /cmd_vel."
    )

    rate = rospy.Rate(rate_hz)
    motion_msg = make_twist_msg(Twist, final_v, final_omega)

    # Give ROS a short moment to connect the publisher before the short test.
    time.sleep(0.5)

    try:
        start_time = time.time()
        while not rospy.is_shutdown() and (time.time() - start_time) < duration:
            publisher.publish(motion_msg)
            rate.sleep()
    except KeyboardInterrupt:
        rospy.logwarn("[test_cmd_vel] Interrupted by user.")
    finally:
        rospy.logwarn("[test_cmd_vel] Sending zero Twist to stop the robot.")
        publish_zero_twist(rospy, publisher, Twist, rate_hz, count=8)

    return 0


if __name__ == "__main__":
    sys.exit(main())
