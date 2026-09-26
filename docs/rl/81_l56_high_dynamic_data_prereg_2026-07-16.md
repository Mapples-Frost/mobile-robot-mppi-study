# L56：固定隐藏物理失配的高动态数据阶段预注册（2026-07-16）

## 1. 固定 plant 与 nominal 边界

L56 所有 episode 使用同一 MuJoCo diff-drive plant：13 kg chassis、固定惯量、低轮地
摩擦、1.55 N·m 扭矩上限、较弱 PI 增益和 40 ms command delay。MPPI predictor 继续使用
原始 dynamic-unicycle 的 0.18/0.12 s 名义时间常数，不读取质量、惯量、摩擦、扭矩或
PI 参数。command delay 作为可测量量仍显式进入 rollout，避免把已知延迟冒充学习贡献。

动作上限提高到 0.65 m/s、1.25 rad/s，rate limit 提高以形成可辨识加减速和转向激励。
场景无障碍，但 LaserScan、scan guard、local obstacle layer 和 safety arbitration 均保持
启用；这样动力学结果不受障碍物真值或关闭安全链影响。

## 2. 路径与数据单元

四条路径分别为 sweep、chicane、acceleration-straight 和 reverse-S。前 3 条用于
train/validation/test，reverse-S 整条路径只进入 unseen-scene split。共 12 个开发 seeds、
48 个 nominal episodes。

分割单位为完整 `(scene, physics_domain, episode_seed)`：

- train：前 3 条路径 × 8 seeds = 24 episodes；
- validation：前 3 条路径 × 1 seed = 3 episodes；
- test：前 3 条路径 × 3 seeds = 9 episodes；
- unseen：reverse-S × 全部 12 seeds = 12 episodes。

四个 split 没有共享 episode；归一化统计只允许由 train 计算。reverse-S 的轨迹形状完全
不进入训练或 checkpoint 选择。

## 3. 数据阶段 Gate

在训练网络前必须确认：

1. 48/48 episode artifact 完整，0 collision；
2. nominal 大部分 episode 成功，若成功率低于 80% 则先降低动态难度；
3. 控制与状态均存在足够跨度，而不是始终饱和或静止；
4. residual target 恒等式和角度 wrap 检查通过；
5. split episode ID 严格不重叠；
6. 数据 metadata 记录固定 plant、路径、seed、git SHA 与 applied-control 语义。

本轮只采 nominal on-policy 数据，不训练 RL、不启用 memory、不使用 learned residual。
若数据 Gate 失败，不进入 ICODE 训练。

### Smoke-stage rejected paths

最初候选的 acceleration-turn 与 self-near hairpin 在 seed 21860731 上出现终点环绕，
延长时限、终点角度门、受限倒车和终端退出段均未达到 80% baseline 可行性要求；这些
尝试分别保留在 `l56_high_dynamic_smoke*` 目录。它们被替换是因为混入了路径/终点收敛
失败，而不是因为 ICODE 结果不好；替换发生在任何 L56 训练和 residual 对照之前。

## 4. Smoke 诊断与观测边界冻结

早期 smoke 继承了 `wheel_odometry`。在低摩擦工况下，轮速里程计误差随打滑累积，
导致规划器观测位置与 MuJoCo 真值不一致；4 个场景仅 2 个成功。该设置会把定位误差
错误地混入动力学 residual，因此不符合 L56“隔离动力学误差”的目标。

在任何正式采数、ICODE 训练或 learned-residual 对照之前，L56 冻结为：

- planner observation 使用 MuJoCo ground-truth pose/twist；
- odom/twist noise 与 latency 均为 0；
- MuJoCo plant 对 planner 仍是隐藏的，planner 不读取质量、惯量、摩擦、PI 或扭矩参数；
- wheel odometry、传感噪声与延迟作为后续独立鲁棒性因子，不在 L56 中混入。

终点局部收敛采用默认关闭的 terminal heading gate 与 terminal alignment law：方位误差
过大时只禁止平移，不禁止角速度；进入成功半径后 episode 立即结束。旧 baseline 未启用
这些开关，因此默认数值行为保持不变。

冻结配置在 seeds 21860731–21860733 的扩大 smoke 上得到 12/12 success、0 collision；
四个场景均为 3/3 success。该结果只用于验证数据采集可行性，不是 ICODE 效果证据。

## 5. 正式数据 Gate 的数值阈值

以下阈值已写入 `configs/rl/icode_high_dynamic_data_l56.yaml`，并在 48-episode 正式采数前冻结：

- episode 数必须为 48，整体 success rate 不低于 0.80；
- 每个场景 success rate 不低于 0.75，collision 必须为 0；
- safety override fraction 必须为 0（本场景无障碍物）；
- applied `v` 的 95% 分位不低于 0.35 m/s；
- applied `|omega|` 的 95% 分位不低于 0.35 rad/s；
- state `v` 标准差不低于 0.08 m/s，state `omega` 标准差不低于 0.10 rad/s；
- issued `v` 上限饱和比例、issued `omega` 双侧饱和比例均不高于 0.25；
- 48 个 resolved config 的 plant/action/sensor contract 必须各自唯一；
- 所有 episode/step key 唯一，禁止 sealed/protected seed 泄漏。

机器可执行检查入口为 `experiments/rl/summarize_high_dynamic_data_gate.py`。只有该 Gate 与
后续 residual dataset identity/split Gate 同时通过，才允许启动 L56 ICODE 训练。
