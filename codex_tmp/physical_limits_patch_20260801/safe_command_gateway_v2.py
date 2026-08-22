#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Robot-side, fail-closed command gateway for laptop RL-MPPI.

Compatible with Ubuntu 16.04 / ROS Kinetic / Python 2.7.  The process cannot
arm unless both --armed and the exact physical-estop confirmation token are
present.  It never accepts a command outside its hard-coded platform caps.
"""

from __future__ import print_function

import argparse
import json
import math
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Range
from std_msgs.msg import Int16, String

from reverse_safety_contract import (
    apply_front_arc_creep,
    apply_directional_clearance,
    bound_signed_speed,
    angular_slew_step_budget,
    limit_angular_command_step,
    motion_session_expired,
    paired_rear_clearance,
    paired_rear_is_fresh,
    verified_near_body_turn_escape,
)


HARD_MAX_LINEAR_MPS = 0.70
HARD_MAX_REVERSE_MPS = 0.70
HARD_MAX_ANGULAR_RADPS = 3.00
ARM_TOKEN = "I_HAVE_PHYSICAL_ESTOP"


def _clamp(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def _yaw(quaternion):
    return math.atan2(
        2.0 * (
            float(quaternion.w) * float(quaternion.z)
            + float(quaternion.x) * float(quaternion.y)
        ),
        1.0
        - 2.0
        * (
            float(quaternion.y) * float(quaternion.y)
            + float(quaternion.z) * float(quaternion.z)
        ),
    )


def _validate_args(args):
    if args.armed and args.confirm_token != ARM_TOKEN:
        raise ValueError(
            "armed mode requires --confirm-token %s" % ARM_TOKEN
        )
    if args.max_linear_mps <= 0.0:
        raise ValueError("max-linear-mps must be positive")
    if args.max_linear_mps > HARD_MAX_LINEAR_MPS:
        raise ValueError(
            "max-linear-mps exceeds hard cap %.3f"
            % HARD_MAX_LINEAR_MPS
        )
    if args.max_reverse_mps < 0.0:
        raise ValueError("max-reverse-mps cannot be negative")
    if args.max_reverse_mps > HARD_MAX_REVERSE_MPS:
        raise ValueError(
            "max-reverse-mps exceeds hard cap %.3f"
            % HARD_MAX_REVERSE_MPS
        )
    if args.max_angular_radps <= 0.0:
        raise ValueError("max-angular-radps must be positive")
    if args.max_angular_radps > HARD_MAX_ANGULAR_RADPS:
        raise ValueError(
            "max-angular-radps exceeds hard cap %.3f"
            % HARD_MAX_ANGULAR_RADPS
        )
    if args.boundary_min_x >= args.boundary_max_x:
        raise ValueError("invalid x boundary")
    if args.boundary_min_y >= args.boundary_max_y:
        raise ValueError("invalid y boundary")
    if (
        args.boundary_min_x + args.boundary_margin_m >= 0.0
        or args.boundary_max_x - args.boundary_margin_m <= 0.0
        or args.boundary_min_y + args.boundary_margin_m >= 0.0
        or args.boundary_max_y - args.boundary_margin_m <= 0.0
    ):
        raise ValueError(
            "boundary must contain the startup origin plus margin"
        )
    if args.command_watchdog_s <= 0.0:
        raise ValueError("command watchdog must be positive")
    if args.angular_slew_radps2 <= 0.0:
        raise ValueError("angular-slew-radps2 must be positive")
    if args.sensor_watchdog_s <= 0.0:
        raise ValueError("sensor watchdog must be positive")
    if args.session_timeout_s <= 0.0:
        raise ValueError("session timeout must be positive")
    if args.front_stop_m <= args.near_body_stop_m:
        raise ValueError("front-stop-m must exceed near-body-stop-m")
    if args.front_slow_m <= args.front_stop_m:
        raise ValueError("front-slow-m must exceed front-stop-m")
    if args.rear_stop_m <= args.near_body_stop_m:
        raise ValueError("rear-stop-m must exceed near-body-stop-m")
    if args.rear_slow_m <= args.rear_stop_m:
        raise ValueError("rear-slow-m must exceed rear-stop-m")


class SafeCommandGatewayV2(object):
    def __init__(self, args):
        self.args = args
        self.started_at = time.time()
        self.motion_session_started_at = None
        self.last_command = None
        self.last_command_time = 0.0
        self.last_applied_angular = 0.0
        self.last_applied_angular_time = None
        self.last_scan_time = 0.0
        self.last_rear_right_time = 0.0
        self.last_rear_left_time = 0.0
        self.last_emergency_time = 0.0
        self.last_odom_time = 0.0
        self.emergency = None
        self.front_clearance = None
        self.rear_clearance = None
        self.rear_right_clearance = None
        self.rear_left_clearance = None
        self.near_body_clearance = None
        self.nearest_obstacle_angle = None
        self.left_turn_clearance = None
        self.right_turn_clearance = None
        self.origin = None
        self.local_pose = None
        self.ever_published_motion = False
        self.motion_publish_count = 0
        self.publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.status_publisher = rospy.Publisher(
            "/mppi_gateway/status", String, queue_size=1, latch=True
        )
        rospy.Subscriber(
            "/mppi_safe_cmd", Twist, self.on_command, queue_size=1
        )
        rospy.Subscriber("/scan", LaserScan, self.on_scan, queue_size=1)
        # The installed RPLIDAR covers only [-90, +90] degrees.  The actual
        # rear coverage is supplied by the two rear-flank ultrasonic sensors:
        # sonar0 at about -105 deg and sonar4 at about +105 deg.
        rospy.Subscriber("/sonar0", Range, self.on_rear_right, queue_size=1)
        rospy.Subscriber("/sonar4", Range, self.on_rear_left, queue_size=1)
        rospy.Subscriber(
            "/emergencybt_status",
            Int16,
            self.on_emergency,
            queue_size=1,
        )
        rospy.Subscriber("/odom", Odometry, self.on_odom, queue_size=1)

    @staticmethod
    def zero_message():
        return Twist()

    def on_command(self, message):
        if self.motion_session_started_at is None:
            # Warm-up and model loading happen before the laptop advertises
            # its first command.  Start the finite motion lease here so that
            # no-publish initialization cannot consume the live session.
            self.motion_session_started_at = time.time()
        self.last_command = message
        self.last_command_time = time.time()

    def on_emergency(self, message):
        self.emergency = int(message.data)
        self.last_emergency_time = time.time()

    def on_odom(self, message):
        pose = message.pose.pose
        if self.origin is None:
            self.origin = (
                float(pose.position.x),
                float(pose.position.y),
                _yaw(pose.orientation),
            )
        x0, y0, yaw0 = self.origin
        dx = float(pose.position.x) - x0
        dy = float(pose.position.y) - y0
        cosine = math.cos(yaw0)
        sine = math.sin(yaw0)
        self.local_pose = (
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            math.atan2(
                math.sin(_yaw(pose.orientation) - yaw0),
                math.cos(_yaw(pose.orientation) - yaw0),
            ),
        )
        self.last_odom_time = time.time()

    def on_scan(self, message):
        front_half = math.radians(float(self.args.front_angle_deg))
        front = []
        near_body = []
        valid_values = []
        left_turn = []
        right_turn = []
        for index, raw_value in enumerate(message.ranges):
            value = float(raw_value)
            if (
                math.isnan(value)
                or math.isinf(value)
                or value < float(message.range_min)
                or value > float(message.range_max)
            ):
                continue
            angle = float(message.angle_min) + (
                index * float(message.angle_increment)
            )
            near_body.append(value)
            valid_values.append((value, angle))
            if abs(angle) <= front_half:
                front.append(value)
            if math.radians(25.0) <= angle <= math.radians(120.0):
                left_turn.append(value)
            if math.radians(-120.0) <= angle <= math.radians(-25.0):
                right_turn.append(value)
        self.front_clearance = min(front) if front else None
        self.near_body_clearance = (
            min(near_body) if near_body else None
        )
        self.nearest_obstacle_angle = (
            min(valid_values, key=lambda item: item[0])[1]
            if valid_values
            else None
        )
        self.left_turn_clearance = min(left_turn) if left_turn else None
        self.right_turn_clearance = min(right_turn) if right_turn else None
        self.last_scan_time = time.time()

    @staticmethod
    def _range_value(message):
        value = float(message.range)
        if math.isnan(value) or math.isinf(value):
            return None
        if value < float(message.min_range) or value > float(message.max_range):
            return None
        return value

    def _update_rear_clearance(self):
        if (
            self.rear_right_clearance is None
            or self.rear_left_clearance is None
        ):
            self.rear_clearance = None
        else:
            self.rear_clearance = paired_rear_clearance(
                self.rear_right_clearance, self.rear_left_clearance
            )

    def on_rear_right(self, message):
        self.rear_right_clearance = self._range_value(message)
        self.last_rear_right_time = time.time()
        self._update_rear_clearance()

    def on_rear_left(self, message):
        self.rear_left_clearance = self._range_value(message)
        self.last_rear_left_time = time.time()
        self._update_rear_clearance()

    def _rear_sensing_fresh(self, now):
        return paired_rear_is_fresh(
            self.rear_right_clearance,
            self.rear_left_clearance,
            now - self.last_rear_right_time,
            now - self.last_rear_left_time,
            self.args.sensor_watchdog_s,
        )

    def publish_zero(self):
        self.publisher.publish(self.zero_message())
        self._record_applied_angular(0.0)

    def _record_applied_angular(self, angular_radps):
        """Bind the slew limiter to the command actually sent to the robot."""
        self.last_applied_angular = float(angular_radps)
        self.last_applied_angular_time = time.monotonic()

    def _inside_boundary(self, x, y):
        margin = float(self.args.boundary_margin_m)
        return (
            float(self.args.boundary_min_x) + margin <= x
            <= float(self.args.boundary_max_x) - margin
            and float(self.args.boundary_min_y) + margin <= y
            <= float(self.args.boundary_max_y) - margin
        )

    def _boundary_accepts_command(self, v):
        if self.local_pose is None:
            return False
        x, y, theta = self.local_pose
        horizon = float(self.args.boundary_prediction_s)
        projected_x = x + float(v) * math.cos(theta) * horizon
        projected_y = y + float(v) * math.sin(theta) * horizon
        return self._inside_boundary(x, y) and self._inside_boundary(
            projected_x, projected_y
        )

    def _decision(self):
        now = time.time()
        if not self.args.armed:
            return None, "disarmed"
        if motion_session_expired(
            self.motion_session_started_at,
            now,
            self.args.session_timeout_s,
        ):
            return self.zero_message(), "session_timeout"
        if self.emergency is None or (
            now - self.last_emergency_time
            > float(self.args.sensor_watchdog_s)
        ):
            return self.zero_message(), "emergency_status_stale"
        if self.emergency != 0:
            return self.zero_message(), "physical_emergency_active"
        if (
            self.front_clearance is None
            or self.near_body_clearance is None
            or now - self.last_scan_time
            > float(self.args.sensor_watchdog_s)
        ):
            return self.zero_message(), "scan_stale_or_invalid"
        if (
            self.local_pose is None
            or now - self.last_odom_time
            > float(self.args.sensor_watchdog_s)
        ):
            return self.zero_message(), "odom_stale_or_invalid"
        if not self._inside_boundary(
            self.local_pose[0], self.local_pose[1]
        ):
            return self.zero_message(), "workspace_boundary"
        if (
            self.last_command is None
            or now - self.last_command_time
            > float(self.args.command_watchdog_s)
        ):
            return self.zero_message(), "command_watchdog"

        requested_v = float(self.last_command.linear.x)
        requested_w = float(self.last_command.angular.z)
        if not self.args.allow_reverse and requested_v < 0.0:
            requested_v = 0.0
        v = bound_signed_speed(
            requested_v,
            self.args.max_linear_mps,
            self.args.max_reverse_mps,
            self.args.allow_reverse,
        )
        w = _clamp(
            requested_w,
            -float(self.args.max_angular_radps),
            float(self.args.max_angular_radps),
        )
        angular_time = time.monotonic()
        angular_elapsed_s = (
            1.0 / float(self.args.rate_hz)
            if self.last_applied_angular_time is None
            else angular_time - self.last_applied_angular_time
        )
        w, _ = limit_angular_command_step(
            w,
            self.last_applied_angular,
            angular_slew_step_budget(
                angular_elapsed_s,
                self.args.angular_slew_radps2,
                1.0 / float(self.args.rate_hz),
            ),
        )
        # The front-only lidar near-body stop must not suppress the planner's
        # escape direction.  Forward/rotation-only commands retain the old
        # hard stop; negative motion is instead governed by both rear sonars.
        if (
            v >= 0.0
            and self.near_body_clearance
            <= float(self.args.near_body_stop_m)
        ):
            escape_w, escape = verified_near_body_turn_escape(
                w,
                self.nearest_obstacle_angle,
                self.left_turn_clearance,
                self.right_turn_clearance,
                self.args.max_angular_radps,
            )
            if escape is None or abs(v) > 1.0e-9:
                return self.zero_message(), "near_body_hard_stop"
            message = Twist()
            message.linear.x = 0.0
            message.angular.z = escape_w
            return message, "near_body_verified_turn_escape"
        if v < 0.0 and not self._rear_sensing_fresh(now):
            return self.zero_message(), "rear_sensors_stale_or_invalid"
        if not self._boundary_accepts_command(v):
            return self.zero_message(), "workspace_boundary_prediction"
        creep_v, creep_reason = apply_front_arc_creep(
            v,
            self.front_clearance,
            self.args.front_stop_m,
            self.args.near_body_stop_m,
        )
        if creep_reason is not None:
            v, reason = creep_v, creep_reason
        else:
            v, reason = apply_directional_clearance(
            v,
            self.front_clearance,
            self.rear_clearance,
            self.args.front_stop_m,
            self.args.front_slow_m,
            self.args.rear_stop_m,
            self.args.rear_slow_m,
            )
        message = Twist()
        message.linear.x = v
        message.angular.z = w
        return message, reason

    def _status(self, reason):
        now = time.time()
        session_age = (
            None
            if self.motion_session_started_at is None
            else now - self.motion_session_started_at
        )
        session_expired = motion_session_expired(
            self.motion_session_started_at,
            now,
            self.args.session_timeout_s,
        )
        return {
            "contract": "rl_mppi_gateway_v2",
            "armed": bool(self.args.armed and not session_expired),
            "reason": reason,
            "emergency": self.emergency,
            "front_clearance_m": self.front_clearance,
            "rear_clearance_m": self.rear_clearance,
            "rear_right_clearance_m": self.rear_right_clearance,
            "rear_left_clearance_m": self.rear_left_clearance,
            "rear_sensor_source": "sonar0_and_sonar4",
            "near_body_clearance_m": self.near_body_clearance,
            "nearest_obstacle_angle_rad": self.nearest_obstacle_angle,
            "left_turn_clearance_m": self.left_turn_clearance,
            "right_turn_clearance_m": self.right_turn_clearance,
            "local_pose": self.local_pose,
            "origin_odom": self.origin,
            "command_age_s": (
                None
                if self.last_command is None
                else now - self.last_command_time
            ),
            "odom_age_s": (
                None
                if self.local_pose is None
                else now - self.last_odom_time
            ),
            "rear_right_age_s": (
                None
                if self.last_rear_right_time <= 0.0
                else now - self.last_rear_right_time
            ),
            "rear_left_age_s": (
                None
                if self.last_rear_left_time <= 0.0
                else now - self.last_rear_left_time
            ),
            "session_started": self.motion_session_started_at is not None,
            "session_age_s": session_age,
            "process_age_s": now - self.started_at,
            "ever_published_motion": self.ever_published_motion,
            "motion_publish_count": self.motion_publish_count,
            "limits": {
                "max_linear_mps": self.args.max_linear_mps,
                "max_reverse_mps": self.args.max_reverse_mps,
                "max_angular_radps": self.args.max_angular_radps,
                "angular_slew_radps2": self.args.angular_slew_radps2,
                "allow_reverse": bool(self.args.allow_reverse),
            },
            "clearance_thresholds": {
                "front_stop_m": self.args.front_stop_m,
                "front_slow_m": self.args.front_slow_m,
                "rear_stop_m": self.args.rear_stop_m,
                "rear_slow_m": self.args.rear_slow_m,
                "near_body_stop_m": self.args.near_body_stop_m,
            },
            "boundary": {
                "min_x": self.args.boundary_min_x,
                "max_x": self.args.boundary_max_x,
                "min_y": self.args.boundary_min_y,
                "max_y": self.args.boundary_max_y,
                "margin_m": self.args.boundary_margin_m,
            },
        }

    def run(self):
        rate = rospy.Rate(float(self.args.rate_hz))
        last_status = 0.0
        while not rospy.is_shutdown():
            message, reason = self._decision()
            if message is not None:
                self.publisher.publish(message)
                self._record_applied_angular(message.angular.z)
                if (
                    abs(float(message.linear.x)) > 1.0e-12
                    or abs(float(message.angular.z)) > 1.0e-12
                ):
                    self.ever_published_motion = True
                    self.motion_publish_count += 1
            now = time.time()
            if now - last_status >= 0.25:
                self.status_publisher.publish(
                    String(
                        data=json.dumps(
                            self._status(reason), sort_keys=True
                        )
                    )
                )
                last_status = now
            rate.sleep()

    def shutdown(self):
        if self.args.armed:
            rate = rospy.Rate(20)
            for _ in range(12):
                self.publish_zero()
                rate.sleep()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--confirm-token", default="")
    parser.add_argument("--allow-reverse", action="store_true")
    parser.add_argument("--max-linear-mps", type=float, default=0.06)
    parser.add_argument("--max-reverse-mps", type=float, default=0.10)
    parser.add_argument("--max-angular-radps", type=float, default=0.25)
    parser.add_argument("--angular-slew-radps2", type=float, default=1.0)
    parser.add_argument("--command-watchdog-s", type=float, default=0.55)
    parser.add_argument("--sensor-watchdog-s", type=float, default=0.40)
    parser.add_argument("--session-timeout-s", type=float, default=180.0)
    parser.add_argument("--front-angle-deg", type=float, default=35.0)
    parser.add_argument("--front-stop-m", type=float, default=0.45)
    parser.add_argument("--front-slow-m", type=float, default=0.80)
    parser.add_argument("--rear-stop-m", type=float, default=0.45)
    parser.add_argument("--rear-slow-m", type=float, default=0.80)
    parser.add_argument("--near-body-stop-m", type=float, default=0.35)
    parser.add_argument("--boundary-min-x", type=float, required=True)
    parser.add_argument("--boundary-max-x", type=float, required=True)
    parser.add_argument("--boundary-min-y", type=float, required=True)
    parser.add_argument("--boundary-max-y", type=float, required=True)
    parser.add_argument("--boundary-margin-m", type=float, default=0.25)
    parser.add_argument("--boundary-prediction-s", type=float, default=0.75)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    args = parser.parse_args()
    _validate_args(args)
    return args


def main():
    args = parse_args()
    rospy.init_node("rl_mppi_command_gateway_v2")
    gateway = SafeCommandGatewayV2(args)
    rospy.on_shutdown(gateway.shutdown)
    rospy.logwarn(
        "[rl_mppi_gateway_v2] armed=%s max_v=%.3f max_w=%.3f"
        % (
            args.armed,
            args.max_linear_mps,
            args.max_angular_radps,
        )
    )
    gateway.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
