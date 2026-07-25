# Online V3 Tracking and Risk-Aware MPPI Smoke 预注册

日期：2026-07-23  
平台：Windows native  
阶段：development smoke  
状态：在线实现与闭环结果产生前冻结

## 1. 目标

验证以下因果链路能否在 MuJoCo 中完整运行：

`LaserScan → 圆形目标检测/关联 → Change-Aware IMM → Collision Risk V1 → MPPI`

本阶段是工程与行为 smoke，不作正式统计优效声明。它不使用 predictor held-out seed，
也不使用 sealed seed。

## 2. 场景

- 机器人起点：`(-3.6, 0.0)`；
- 机器人终点：`(3.6, 0.0)`；
- 机器人直接路线是水平走廊 `y=0`；
- 障碍物采用冻结 V3 `hybrid_patrol`；
- V3 航点同时分布于走廊上下方，障碍物会在巡逻和随机干预过程中多次穿越机器人
  必经路线；
- 单 episode 最长 40 s，短于冻结 V3 45 s 轨迹，因此不在首尾拼接处循环或瞬移。

障碍物的 V3 seed、状态、模式和未来轨迹只属于 MuJoCo 真值与离线审计。控制器只能
使用带噪 LaserScan、里程计和由历史观测产生的预测。

## 3. 在线检测与关联契约

1. 将当前 LaserScan 的有效点变换到世界坐标；
2. 按相邻 beam 和相邻点距离形成圆弧 cluster；
3. 使用已知实验障碍半径，从最近表面 beam 恢复圆心观测；
4. tracker 未初始化时选择支持 beam 最多的 cluster；
5. tracker 已初始化时，选择距 IMM 一步预测最近且位于 0.90 m gate 内的 cluster；
6. 未关联时向 IMM 输入 `None`，由冻结 dropout guard 扩大不确定性；
7. 连续未观测超过 1.50 s 后不再输出有效 forecast，系统必须 fail closed；
8. 输出与当前 observation 同时间戳、与 MPPI 同为 0.10 s 步长的 36 步混合高斯预测。

接口不得接收 MuJoCo truth、V3 mode、change flag、trajectory seed 或未来障碍状态。

## 4. Paired smoke 设计

- development seeds：`730100101–730100103`；
- 两个 arm：`risk_disabled`、`risk_enabled`；
- 每个 seed 的两个 arm 共享完整 V3 轨迹、传感器噪声和 MPPI 随机数；
- 运行顺序由 seed `730199988` 随机打乱；
- 独立单位是完整 episode，不把 episode 内控制步当作独立重复；
- risk-disabled 仍运行同一个 tracker/predictor，只关闭 MPPI 风险代价，以隔离风险代价
  的作用。

## 5. Smoke Gate

必须满足：

1. 每个 episode 的障碍物中心至少 3 次穿越 `y=0` 走廊；
2. tracker forecast 可用率至少 90%；
3. 障碍物位于可见范围时，圆心测量 RMSE 不超过 0.12 m；
4. 所有 forecast 有限、权重有效、协方差对称且 PSD；
5. 无任何真值泄漏接口；
6. risk-enabled 的碰撞次数不高于配对 risk-disabled；
7. risk-enabled 的完成进度下降不超过 0.10；
8. risk-enabled 的中位最小 clearance 差不低于 -0.02 m；
9. 至少一对 seed 的控制或轨迹产生可测行为差异；
10. risk-enabled 端到端 planner P95 `compute_ms` 不超过 150 ms；
11. 所有运行和汇总产物具有配置、环境、seed、源码哈希和唯一运行键。

Gate 失败时允许在 development seed 上诊断和修订在线检测或控制参数，但不得修改冻结
V3、Change-Aware IMM 或 Collision Risk V1 数学定义。每次修订必须形成新 amendment。

## 6. 解释限制

- 单目标已知半径关联是进入闭环的最小科研可解释版本，不代表多目标数据关联已解决；
- 当前场景没有静态障碍与动态障碍身份混淆，后续多目标实验需要 JPDA/GNN 等单独阶段；
- smoke 通过仅表示链路工作且有合理行为，不表示正式安全优效；
- 正式闭环实验仍需新 held-out episode seed、预先确定的主指标和功效分析。

