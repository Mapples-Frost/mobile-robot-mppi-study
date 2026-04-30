# E1/E2 小车五一实验室运行手册

这份手册用于第一次把当前 MPPI 硬件桥接层接到真实 E1/E2 小车。目标是先验证 `/cmd_vel`、`/odom`、`/scan` 和安全兜底，不是第一次就接完整 MPPI。

## 0. 实验前安全原则

1. 全程低速。第一次速度建议 `0.03 m/s`，角速度保持很小。
2. `mppi_hardware_bridge/config/lab_runtime.yaml` 里的 `safety.enable_publish` 默认必须是 `false`。
3. 不要让两个节点同时发布 `/cmd_vel`。运行控制脚本前必须检查 `rostopic info /cmd_vel`。
4. 先单独测试 `/cmd_vel`、`/odom`、`/scan`，再跑 adapter。
5. 第一次不要接完整 MPPI。先用 `mppi_ros_adapter_skeleton.py` dry-run 看状态链路。
6. 先 dry-run，再 enable publish。确认安全后才允许修改配置或用命令行打开发布。
7. 现场必须有人看车，手边能急停，必要时直接断开驱动或停节点。

## 1. 到实验室第一步

先确认网络和机器人 SSH：

```bash
ping 192.168.31.200
ssh eaibot@192.168.31.200
```

进入机器人或 ROS 机器后检查 ROS：

```bash
rosversion -d
rostopic list
```

如果 `rostopic list` 卡住或报 master 相关错误，先检查 `ROS_MASTER_URI`、`ROS_IP`、网络和机器人上的 roscore/bringup。

## 2. 检查关键 topic

重点看这些 topic 是否存在：

```bash
rostopic list | grep cmd_vel
rostopic list | grep odom
rostopic list | grep scan
```

需要特别确认：

- `/cmd_vel`
- `/smoother_cmd_vel`
- `/odom`
- `/robot_pose_ekf/odom_combined`
- `/scan`

检查 `/cmd_vel` 是否已经有其他 publisher：

```bash
rostopic info /cmd_vel
```

如果已经看到 `move_base`、速度平滑器或其他控制节点在发布 `/cmd_vel`，不能同时运行自己的控制脚本。先停掉旧控制器，或改成现场确认后的唯一控制通道。

## 3. 单项测试顺序

按这个顺序跑，不要跳步：

1. `test_cmd_vel.py`
2. `echo_odom.py`
3. `test_scan_guard_ros.py`
4. `mppi_ros_adapter_skeleton.py` dry-run
5. 确认安全后，再 enable publish

## 4. `/cmd_vel` 低速测试

先 dry-run：

```bash
python mppi_hardware_bridge/scripts/test_cmd_vel.py
```

确认打印的 topic、速度、duration 都正确后，再极低速发布：

```bash
python mppi_hardware_bridge/scripts/test_cmd_vel.py --enable --linear 0.03 --angular 0.0 --duration 2.0
```

这个脚本只测 `/cmd_vel`，不接 MPPI、不接 scan、不接 odom。duration 结束后会自动连续发布 zero Twist 停车。

手动 zero Twist 停车命令：

```bash
rostopic pub -1 /cmd_vel geometry_msgs/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
```

如果车不动，检查：

- `/cmd_vel` 是否是驱动真正监听的 topic。
- 是否实际需要发到 `/smoother_cmd_vel`。
- driver/bringup 是否启动。
- 急停、底盘电源、电机使能是否打开。
- `rostopic echo /cmd_vel` 是否能看到 Twist。

## 5. `/odom` 和实验坐标系测试

默认读 `cfg.ros.odom_topic`：

```bash
python mppi_hardware_bridge/scripts/echo_odom.py
```

如果默认 odom 没数据，用 fallback：

```bash
python mppi_hardware_bridge/scripts/echo_odom.py --use-fallback-odom
```

观察输出里的两组数据：

- `raw odom`: 机器人原始 odom 坐标。
- `experiment frame`: 以脚本启动第一帧为原点、启动朝向为 +x 的实验坐标。

判断方法：

- 小车从启动方向向前走时，`x_exp` 应主要增加。
- 小车左右偏移时，`y_exp` 应变化。
- 小车原地转动时，`yaw_exp` 应变化。

如果 `x_exp/y_exp` 方向不符合直觉，先不要继续跑 adapter，先确认 odom frame、机器人前进方向和 `frame_transform.py` 的相对坐标约定。

## 6. `/scan` 和 scan_guard 测试

运行：

```bash
python mppi_hardware_bridge/scripts/test_scan_guard_ros.py
```

用纸箱慢慢靠近小车正前方，应该依次看到：

- `front_clear`
- `front_obstacle_slow`
- `front_obstacle_stop`

如果纸箱在侧边却触发 stop，可能 `safety.front_angle_deg` 太大。

如果正前方障碍不触发，可能雷达的正前方角度不是 `angle=0`，需要调整：

```bash
python mppi_hardware_bridge/scripts/test_scan_guard_ros.py --front-angle-offset-deg <角度>
```

调角度时一次只改小步，比如 `5`、`10`、`-5`、`-10`，观察 `min_front_range` 和 `reason`。

## 7. Adapter skeleton dry-run

第一次只 dry-run：

```bash
python mppi_hardware_bridge/scripts/mppi_ros_adapter_skeleton.py
```

默认 `safety.enable_publish=false`，脚本不会发布 `/cmd_vel`。它会订阅 odom/scan/goal，并打印：

- 当前 experiment frame 状态。
- 当前 goal。
- fake planner 的 `(v, omega)`。
- scan_guard 结果。
- 经过 `control_adapter` 后的最终 `(v, omega)`。

注意：这里的 `fake_planner_control(...)` 只是 adapter 骨架测试，不是最终 MPPI。

确认下面几件事都正常后，才允许 enable publish：

- odom 有数据。
- scan 有数据。
- 纸箱靠近时 emergency_stop 会变 true。
- goal 坐标合理。
- final control 没有超过 `limits.v_max` 和 `limits.w_max`。
- `rostopic info /cmd_vel` 没有其他控制器抢 topic。

现场确认安全后，可以用命令行临时打开发布：

```bash
python mppi_hardware_bridge/scripts/mppi_ros_adapter_skeleton.py --enable-publish
```

或者修改 `lab_runtime.yaml` 里的 `safety.enable_publish: true`，但实验结束后建议改回 `false`。

## 8. 常见故障排查

没有 `rospy`：

- 说明当前 shell 没有 ROS 环境。
- 执行 `source /opt/ros/<distro>/setup.bash`。
- 如果有 catkin workspace，再执行 `source <your_catkin_ws>/devel/setup.bash`。

没有 `/scan`：

- 检查雷达节点是否启动。
- `rostopic list | grep scan` 看真实 topic 名。
- `rostopic echo /scan` 看是否有 LaserScan 数据。

没有 `/odom`：

- 先试 `/robot_pose_ekf/odom_combined`。
- 再试 `/odom`。
- 检查底盘 driver、EKF、TF 是否正常。

`/cmd_vel` 有其他 publisher：

- 不要运行自己的控制脚本。
- 先停 `move_base` 或其他控制节点。
- 再确认 `rostopic info /cmd_vel` 只剩一个预期 publisher。

雷达方向不对：

- 用纸箱分别放在正前、左侧、右侧。
- 如果正前方不触发，调 `--front-angle-offset-deg`。
- 如果侧边触发过多，减小 `safety.front_angle_deg`。

程序一启动就停：

- 可能没有 scan，按 fail-safe 会 emergency stop。
- 可能正前方太近。
- 可能 `front_stop_distance` 太大。

车乱转：

- 先停脚本并发布 zero Twist。
- 检查 odom yaw 是否跳变。
- 检查 goal 是否在错误方向。
- 降低 `limits.w_max`。

车不动：

- 确认用了 `--enable` 或 `--enable-publish`。
- 确认 `safety.enable_publish` 是否仍是 false。
- 检查 `/cmd_vel` 是否被 driver 监听。
- 检查急停、电机电源、底盘使能。

网络连不上：

- `ping 192.168.31.200`。
- 检查电脑和小车是否在同一网段。
- 检查机器人 IP 是否变化。

driver 没启动或重复启动：

- 没启动时没有 odom/cmd_vel 响应。
- 重复启动时可能多个节点抢底盘或 topic。
- 先 `rosnode list`，再决定停哪个节点。

## 9. 实验记录模板

每次跑真实车都记录：

- 日期：
- 场地：
- 分支 / commit：
- 配置文件：
- 速度限制：
- goal：
- boundary：
- 是否触发急停：
- 是否成功：
- 失败原因：
- 视频文件名：
- rosbag 文件名：
- 备注：

建议每次修改配置后都记录一条，尤其是 `front_angle_offset_deg`、`front_stop_distance`、`v_max`、`w_max` 和 goal。
