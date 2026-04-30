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

from frame_transform import ExperimentFrameTransform
from frame_transform import normalize_angle


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


def quaternion_to_yaw(qx, qy, qz, qw):
    qx = float(qx)
    qy = float(qy)
    qz = float(qz)
    qw = float(qw)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return normalize_angle(math.atan2(siny_cosp, cosy_cosp))


class OdomEcho(object):
    def __init__(self, rospy, odom_type, cfg, use_fallback_odom, print_period):
        self.rospy = rospy
        self.cfg = cfg
        self.print_period = max(0.05, float(print_period))
        self.last_print_time = 0.0
        self.received_count = 0
        self.frame_tf = ExperimentFrameTransform()

        if use_fallback_odom:
            self.odom_topic = cfg.ros.fallback_odom_topic
        else:
            self.odom_topic = cfg.ros.odom_topic

        self.odom_sub = rospy.Subscriber(
            self.odom_topic,
            odom_type,
            self.odom_callback,
            queue_size=1,
        )

        rospy.loginfo("[echo_odom] Subscribed to odom topic: {}".format(self.odom_topic))

    def odom_callback(self, msg):
        self.received_count += 1

        pose = msg.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        q = pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)

        if not self.frame_tf.has_origin:
            self.frame_tf.set_origin(x, y, yaw)
            self.rospy.loginfo(
                "[echo_odom] Set experiment origin from first odom frame: "
                "x={:.3f}, y={:.3f}, yaw={:.3f}".format(x, y, yaw)
            )

        x_exp, y_exp, yaw_exp = self.frame_tf.odom_to_experiment(x, y, yaw)

        now = time.time()
        if now - self.last_print_time < self.print_period:
            return

        self.last_print_time = now

        self.rospy.loginfo(
            "\n[echo_odom] count={count}\n"
            "raw odom:\n"
            "  x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}\n"
            "experiment frame:\n"
            "  x_exp={x_exp:.3f}, y_exp={y_exp:.3f}, yaw_exp={yaw_exp:.3f}".format(
                count=self.received_count,
                x=x,
                y=y,
                yaw=yaw,
                x_exp=x_exp,
                y_exp=y_exp,
                yaw_exp=yaw_exp,
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Echo odometry and relative_to_start experiment-frame pose."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to lab_runtime.yaml.",
    )
    parser.add_argument(
        "--use-fallback-odom",
        action="store_true",
        help="Subscribe to cfg.ros.fallback_odom_topic instead of cfg.ros.odom_topic.",
    )
    parser.add_argument(
        "--print-period",
        type=float,
        default=0.5,
        help="Print interval in seconds. Default: 0.5.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import rospy
        from nav_msgs.msg import Odometry
    except ImportError as exc:
        print_ros_import_error(exc)
        return 1

    try:
        from scenario_config import load_lab_runtime_config
        cfg = load_lab_runtime_config(args.config)
    except Exception as exc:
        print_config_error(exc)
        return 1

    rospy.init_node("echo_odom", anonymous=True)

    OdomEcho(
        rospy=rospy,
        odom_type=Odometry,
        cfg=cfg,
        use_fallback_odom=args.use_fallback_odom,
        print_period=args.print_period,
    )

    rospy.loginfo("[echo_odom] Waiting for Odometry messages.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logwarn("[echo_odom] Interrupted by user.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
