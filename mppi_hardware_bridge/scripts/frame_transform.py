from __future__ import print_function

import math


def normalize_angle(angle):
    """
    把角度归一化到 [-pi, pi]。

    angle:
        输入角度，单位是 rad。

    返回：
        归一化后的角度，单位还是 rad。
    """
    return math.atan2(math.sin(angle), math.cos(angle))


class ExperimentFrameTransform(object):
    """
    把原始 odom 坐标转换成实验坐标系。

    实验坐标系定义：
        adapter 启动时的小车位置 = (0, 0)
        adapter 启动时的小车车头方向 = +x
        adapter 启动时的小车左侧方向 = +y

    这个类不依赖 ROS，所以现在可以先本地测试。
    """

    def __init__(self):
        self.has_origin = False

        self.x0 = 0.0
        self.y0 = 0.0
        self.yaw0 = 0.0

        self.cos0 = 1.0
        self.sin0 = 0.0

    def set_origin(self, x_odom, y_odom, yaw_odom):
        """
        设置实验坐标系原点。

        x_odom, y_odom:
            小车第一帧 odom 里的位置，单位 m。

        yaw_odom:
            小车第一帧 odom 里的朝向，单位 rad。
        """
        self.x0 = float(x_odom)
        self.y0 = float(y_odom)
        self.yaw0 = float(yaw_odom)

        self.cos0 = math.cos(self.yaw0)
        self.sin0 = math.sin(self.yaw0)

        self.has_origin = True

    def odom_to_experiment(self, x_odom, y_odom, yaw_odom):
        """
        把 odom 坐标转换成 experiment frame 坐标。

        输入：
            x_odom, y_odom, yaw_odom

        输出：
            x_exp, y_exp, yaw_exp

        其中：
            x_exp 表示相对启动点的前方距离
            y_exp 表示相对启动点的左侧距离
            yaw_exp 表示相对启动时车头方向的角度差
        """
        if not self.has_origin:
            raise RuntimeError("Experiment frame origin has not been set.")

        dx = float(x_odom) - self.x0
        dy = float(y_odom) - self.y0

        x_exp = self.cos0 * dx + self.sin0 * dy
        y_exp = -self.sin0 * dx + self.cos0 * dy
        yaw_exp = normalize_angle(float(yaw_odom) - self.yaw0)

        return x_exp, y_exp, yaw_exp

    def experiment_to_odom(self, x_exp, y_exp, yaw_exp=0.0):
        """
        把 experiment frame 坐标转回 odom 坐标。

        后面如果 RViz / debug 需要把实验坐标下的 goal 显示回 odom，
        可以用这个函数。

        输入：
            x_exp, y_exp, yaw_exp

        输出：
            x_odom, y_odom, yaw_odom
        """
        if not self.has_origin:
            raise RuntimeError("Experiment frame origin has not been set.")

        x_exp = float(x_exp)
        y_exp = float(y_exp)

        x_odom = self.x0 + self.cos0 * x_exp - self.sin0 * y_exp
        y_odom = self.y0 + self.sin0 * x_exp + self.cos0 * y_exp
        yaw_odom = normalize_angle(float(yaw_exp) + self.yaw0)

        return x_odom, y_odom, yaw_odom


def _run_basic_tests():
    """
    本地小测试。

    这里不需要 ROS，只检查坐标变换公式是否符合直觉。
    """

    tf = ExperimentFrameTransform()

    # 假设小车启动时在 odom 坐标 (10, -3)，车头朝 odom 的 +y 方向。
    # yaw = pi / 2 表示朝 +y。
    tf.set_origin(10.0, -3.0, math.pi / 2.0)

    # 第一帧自己应该变成 experiment frame 的原点。
    x_exp, y_exp, yaw_exp = tf.odom_to_experiment(10.0, -3.0, math.pi / 2.0)

    print("Test 1: origin")
    print("  x_exp = {:.3f}, y_exp = {:.3f}, yaw_exp = {:.3f}".format(
        x_exp, y_exp, yaw_exp
    ))

    # odom 中从 (10, -3) 到 (10, -2)，是沿 odom +y 走了 1m。
    # 但小车启动时车头正好朝 odom +y，
    # 所以在 experiment frame 里应该是 x_exp = 1, y_exp = 0。
    x_exp, y_exp, yaw_exp = tf.odom_to_experiment(10.0, -2.0, math.pi / 2.0)

    print("Test 2: move forward 1m in initial heading")
    print("  x_exp = {:.3f}, y_exp = {:.3f}, yaw_exp = {:.3f}".format(
        x_exp, y_exp, yaw_exp
    ))

    # odom 中从 (10, -3) 到 (9, -3)，是沿 odom -x 走了 1m。
    # 当初始车头朝 odom +y 时，左侧方向就是 odom -x。
    # 所以在 experiment frame 里应该是 x_exp = 0, y_exp = 1。
    x_exp, y_exp, yaw_exp = tf.odom_to_experiment(9.0, -3.0, math.pi / 2.0)

    print("Test 3: move left 1m relative to initial heading")
    print("  x_exp = {:.3f}, y_exp = {:.3f}, yaw_exp = {:.3f}".format(
        x_exp, y_exp, yaw_exp
    ))

    # 再测一下反变换。
    x_odom, y_odom, yaw_odom = tf.experiment_to_odom(1.0, 0.0, 0.0)

    print("Test 4: experiment goal (1, 0) back to odom")
    print("  x_odom = {:.3f}, y_odom = {:.3f}, yaw_odom = {:.3f}".format(
        x_odom, y_odom, yaw_odom
    ))


if __name__ == "__main__":
    _run_basic_tests()