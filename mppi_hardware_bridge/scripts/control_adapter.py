from __future__ import print_function


def clip(value, lower, upper):
    """
    把 value 限制在 [lower, upper] 之间。

    例子：
        clip(2.0, 0.0, 1.0) -> 1.0
        clip(-1.0, 0.0, 1.0) -> 0.0
        clip(0.5, 0.0, 1.0) -> 0.5
    """
    return max(lower, min(upper, value))


def clamp_mppi_control(
    control,
    v_max,
    w_max,
    allow_backward=False,
    slow_scale=1.0,
):
    """
    把 MPPI 输出的控制量裁剪成真实小车安全可执行的控制量。

    输入：
        control:
            MPPI 输出的控制量，格式是 (v, omega)。

            v:
                线速度，单位 m/s。

            omega:
                角速度，单位 rad/s。

        v_max:
            真实小车允许的最大线速度，单位 m/s。

        w_max:
            真实小车允许的最大角速度，单位 rad/s。

        allow_backward:
            是否允许倒车。
            第一版硬件实验建议 False，也就是不倒车。

        slow_scale:
            速度缩放系数。
            正常是 1.0。
            如果前方有障碍但还没到急停距离，可以设成 0.3、0.5 做减速。

    输出：
        safe_control:
            裁剪后的 (v, omega)。

        debug:
            调试信息，方便后面打印。
    """
    v, omega = control

    v = float(v)
    omega = float(omega)

    v_max = float(v_max)
    w_max = float(w_max)
    slow_scale = clip(float(slow_scale), 0.0, 1.0)

    raw_v = v
    raw_omega = omega

    if allow_backward:
        v = clip(v, -v_max, v_max)
    else:
        v = clip(v, 0.0, v_max)

    omega = clip(omega, -w_max, w_max)

    v_before_slow = v
    v = v * slow_scale

    debug = {
        "raw_v": raw_v,
        "raw_omega": raw_omega,
        "clipped_v_before_slow": v_before_slow,
        "clipped_v": v,
        "clipped_omega": omega,
        "v_max": v_max,
        "w_max": w_max,
        "allow_backward": allow_backward,
        "slow_scale": slow_scale,
    }

    return (v, omega), debug


def apply_emergency_stop(control, emergency_stop=False, reason=""):
    """
    根据 emergency_stop 标志决定是否强制停车。

    输入：
        control:
            当前准备执行的 (v, omega)。

        emergency_stop:
            True 表示必须停车。

        reason:
            停车原因，用于 debug，例如 "front_obstacle"。

    输出：
        final_control:
            如果 emergency_stop=True，返回 (0.0, 0.0)。
            否则返回原 control。

        debug:
            停车调试信息。
    """
    if emergency_stop:
        debug = {
            "emergency_stop": True,
            "reason": reason,
            "input_v": float(control[0]),
            "input_omega": float(control[1]),
            "output_v": 0.0,
            "output_omega": 0.0,
        }
        return (0.0, 0.0), debug

    debug = {
        "emergency_stop": False,
        "reason": reason,
        "input_v": float(control[0]),
        "input_omega": float(control[1]),
        "output_v": float(control[0]),
        "output_omega": float(control[1]),
    }
    return control, debug


def control_to_twist_dict(control):
    """
    把 (v, omega) 转成一个类似 ROS Twist 的普通 dict。

    现在这个文件不直接 import rospy / geometry_msgs，
    这样你在没有 ROS 的本地 PyCharm 里也能测试。

    后面真正写 mppi_ros_adapter.py 时，会把它变成：
        msg = geometry_msgs.msg.Twist()
        msg.linear.x = v
        msg.angular.z = omega
    """
    v, omega = control

    return {
        "linear": {
            "x": float(v),
            "y": 0.0,
            "z": 0.0,
        },
        "angular": {
            "x": 0.0,
            "y": 0.0,
            "z": float(omega),
        },
    }


def prepare_safe_command(
    proposed_control,
    v_max,
    w_max,
    allow_backward=False,
    slow_scale=1.0,
    emergency_stop=False,
    emergency_reason="",
):
    """
    一站式处理函数。

    输入 MPPI proposed_control = (v, omega)，输出安全后的控制和 Twist dict。

    处理顺序：
        1. 先按 v_max / w_max 裁剪；
        2. 再根据 slow_scale 减速；
        3. 最后如果 emergency_stop=True，强制变成 (0.0, 0.0)。

    这个顺序很重要：
        emergency stop 永远最后兜底。
    """
    clipped_control, clamp_debug = clamp_mppi_control(
        control=proposed_control,
        v_max=v_max,
        w_max=w_max,
        allow_backward=allow_backward,
        slow_scale=slow_scale,
    )

    final_control, stop_debug = apply_emergency_stop(
        control=clipped_control,
        emergency_stop=emergency_stop,
        reason=emergency_reason,
    )

    twist_dict = control_to_twist_dict(final_control)

    debug = {
        "clamp": clamp_debug,
        "stop": stop_debug,
        "twist": twist_dict,
    }

    return final_control, twist_dict, debug


def _run_basic_tests():
    """
    本地测试，不依赖 ROS，不依赖 MPPI。
    """

    v_max = 0.08
    w_max = 0.30

    print("Test 1: normal MPPI control within limits")
    final_control, twist, debug = prepare_safe_command(
        proposed_control=(0.05, 0.10),
        v_max=v_max,
        w_max=w_max,
    )
    print("  final_control =", final_control)
    print("  twist =", twist)

    print("")
    print("Test 2: MPPI control exceeds hardware limits")
    final_control, twist, debug = prepare_safe_command(
        proposed_control=(1.20, 2.00),
        v_max=v_max,
        w_max=w_max,
    )
    print("  final_control =", final_control)
    print("  twist =", twist)

    print("")
    print("Test 3: negative v, backward disabled")
    final_control, twist, debug = prepare_safe_command(
        proposed_control=(-0.20, -0.10),
        v_max=v_max,
        w_max=w_max,
        allow_backward=False,
    )
    print("  final_control =", final_control)
    print("  twist =", twist)

    print("")
    print("Test 4: slow_scale = 0.5")
    final_control, twist, debug = prepare_safe_command(
        proposed_control=(0.08, 0.10),
        v_max=v_max,
        w_max=w_max,
        slow_scale=0.5,
    )
    print("  final_control =", final_control)
    print("  twist =", twist)

    print("")
    print("Test 5: emergency stop")
    final_control, twist, debug = prepare_safe_command(
        proposed_control=(0.05, 0.10),
        v_max=v_max,
        w_max=w_max,
        emergency_stop=True,
        emergency_reason="front_obstacle",
    )
    print("  final_control =", final_control)
    print("  twist =", twist)
    print("  stop_debug =", debug["stop"])


if __name__ == "__main__":
    _run_basic_tests()