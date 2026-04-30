# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from scan_guard import analyze_scan_front_sector


DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "config",
        "lab_runtime.yaml",
    )
)


def print_config_error(exc):
    print("")
    print("Config load failed.")
    print("Please check lab_runtime.yaml and Python yaml support.")
    print("Original error: {}".format(exc))
    print("")


class ScanGuardRosTester(object):
    """
    ROS 版 scan_guard 测试器。

    它只做三件事：
        1. 订阅 /scan；
        2. 把真实 LaserScan 消息传给 analyze_scan_front_sector；
        3. 打印前方安全状态。

    它不发 /cmd_vel，不控制小车。
    """

    def __init__(
        self,
        rospy,
        laser_scan_type,
        cfg,
        front_angle_offset_deg=0.0,
        print_period=0.5,
    ):
        self.rospy = rospy
        self.cfg = cfg
        self.front_angle_offset_deg = float(front_angle_offset_deg)
        self.print_period = float(print_period)

        self.last_print_time = 0.0
        self.received_count = 0

        self.scan_sub = rospy.Subscriber(
            cfg.ros.scan_topic,
            laser_scan_type,
            self.scan_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "[test_scan_guard_ros] Subscribed to scan topic: {}".format(
                cfg.ros.scan_topic
            )
        )
        rospy.loginfo(
            "[test_scan_guard_ros] front_stop_distance = {:.3f} m, "
            "front_slow_distance = {:.3f} m, front_angle = {:.1f} deg".format(
                cfg.safety.front_stop_distance,
                cfg.safety.front_slow_distance,
                cfg.safety.front_angle_deg,
            )
        )

    def scan_callback(self, scan_msg):
        """
        每收到一帧 /scan，ROS 会自动调用这个函数。

        scan_msg 是 sensor_msgs/LaserScan 类型。
        里面包含：
            ranges
            angle_min
            angle_increment
            range_min
            range_max
        """
        self.received_count += 1

        result = analyze_scan_front_sector(
            ranges=scan_msg.ranges,
            angle_min=scan_msg.angle_min,
            angle_increment=scan_msg.angle_increment,
            range_min=scan_msg.range_min,
            range_max=scan_msg.range_max,
            front_stop_distance=self.cfg.safety.front_stop_distance,
            front_slow_distance=self.cfg.safety.front_slow_distance,
            front_angle_deg=self.cfg.safety.front_angle_deg,
            front_angle_offset_deg=self.front_angle_offset_deg,
        )

        now = time.time()
        if now - self.last_print_time < self.print_period:
            return

        self.last_print_time = now

        min_front_range = result["min_front_range"]
        if min_front_range is None:
            min_range_text = "None"
        else:
            min_range_text = "{:.3f} m".format(min_front_range)

        msg = (
            "[scan_guard] count={count} | reason={reason} | "
            "emergency_stop={emergency_stop} | slow={slow} | "
            "slow_scale={slow_scale:.3f} | min_front_range={min_range} | "
            "valid_front_count={valid_count}"
        ).format(
            count=self.received_count,
            reason=result["reason"],
            emergency_stop=result["emergency_stop"],
            slow=result["should_slow_down"],
            slow_scale=result["slow_scale"],
            min_range=min_range_text,
            valid_count=result["valid_front_count"],
        )

        if result["emergency_stop"]:
            self.rospy.logwarn(msg)
        elif result["should_slow_down"]:
            self.rospy.loginfo(msg)
        else:
            self.rospy.loginfo(msg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to lab_runtime.yaml",
    )

    parser.add_argument(
        "--front-angle-offset-deg",
        type=float,
        default=0.0,
        help=(
            "Laser front direction offset in degrees. "
            "Default assumes LaserScan angle=0 is robot front."
        ),
    )

    parser.add_argument(
        "--print-period",
        type=float,
        default=0.5,
        help="Print interval in seconds.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import rospy
        from sensor_msgs.msg import LaserScan
    except ImportError as exc:
        print("")
        print("ROS import failed.")
        print("This script must be run in a ROS environment.")
        print("Original error: {}".format(exc))
        print("")
        print("On the robot or ROS machine, run something like:")
        print("  source /opt/ros/<distro>/setup.bash")
        print("  source <your_catkin_ws>/devel/setup.bash")
        print("")
        return 1

    try:
        from scenario_config import load_lab_runtime_config
        cfg = load_lab_runtime_config(args.config)
    except Exception as exc:
        print_config_error(exc)
        return 1

    rospy.init_node("test_scan_guard_ros", anonymous=True)

    tester = ScanGuardRosTester(
        rospy=rospy,
        laser_scan_type=LaserScan,
        cfg=cfg,
        front_angle_offset_deg=args.front_angle_offset_deg,
        print_period=args.print_period,
    )

    rospy.loginfo(
        "[test_scan_guard_ros] Waiting for LaserScan messages. "
        "Move a box or hand slowly in front of the robot to test SAFE/SLOW/STOP."
    )

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logwarn("[test_scan_guard_ros] Interrupted by user.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
