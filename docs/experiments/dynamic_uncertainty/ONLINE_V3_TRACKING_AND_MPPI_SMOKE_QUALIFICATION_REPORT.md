# Online V3 Tracking and Risk-Aware MPPI Smoke Qualification Report

日期：2026-07-23  
平台：Windows native / MuJoCo  
最终配置：`mujoco_v3_probabilistic_crossing_smoke_amendment5.yaml`  
结论：**development smoke Gate 通过（9/9）**

## 1. 本阶段合格结论

以下完整因果链已经在真实 MuJoCo 闭环中运行：

`LaserScan → 单目标检测/预测门控关联 → Change-Aware IMM →`
`Collision Risk V1 → MPPI`

控制器没有接收 V3 seed、mode、change flag、未来轨迹或 MuJoCo 障碍物真值。
真值只在完整 episode 结束后用于路线穿越计数和测量误差审计。

本结果说明环境、在线概率预测接口和风险控制接口已经具备进入后续算法实验
的资格。它不构成“risk-aware MPPI 已经正式优于基线”的统计结论。

## 2. Paired 设计

- 独立单位：完整 episode；
- development seeds：`730100001, 730100003, 730100005`；
- paired arms：`risk_disabled, risk_enabled`；
- 相同 seed 的两臂共享冻结 V3 轨迹、传感器随机流和 MPPI 随机种子；
- 运行顺序由 schedule seed `730199988` 预先随机化；
- 每个 episode 最长 40 s，共 6 个完整 episode。

## 3. Gate 结果

| Gate | 阈值 | 结果 | 判定 |
|---|---:|---:|---|
| 每个 40 s episode 的路线穿越次数 | ≥ 3 | 最低 3 | PASS |
| tracker forecast availability | ≥ 0.90 | 最低 1.00 | PASS |
| 可见时圆心测量 RMSE | ≤ 0.12 m | 最差 0.0908 m | PASS |
| forecast/诊断契约 | 6/6 | 6/6 | PASS |
| risk-enabled 新增碰撞 | ≤ 0 | 0 | PASS |
| risk-enabled 最大完成度下降 | ≤ 0.10 | 0.0103 | PASS |
| risk-enabled 中位最小 clearance 差 | ≥ -0.02 m | +0.1376 m | PASS |
| 行为发生差异的 paired seed | ≥ 1 | 3 | PASS |
| risk-enabled planner P95 | ≤ 150 ms | 最坏 37.65 ms | PASS |

三条冻结 V3 轨迹在 40 s 内分别穿越起终点直线路线 5、3、4 次；障碍物
不是只运动一次，也不是固定周期的简单直线往复。

## 4. 描述性闭环结果

| Seed | disabled 碰撞 | enabled 碰撞 | disabled 完成度 | enabled 完成度 | clearance 差 |
|---|---:|---:|---:|---:|---:|
| 730100001 | 1 | 1 | 0.459 | 0.448 | -0.003 m |
| 730100003 | 1 | 0 | 0.471 | 0.627 | +0.138 m |
| 730100005 | 1 | 0 | 0.393 | 0.695 | +0.170 m |

在这三个 development seed 上，risk-disabled 为 3/3 碰撞，risk-enabled
为 1/3 碰撞。样本量太小且 seed 已用于开发诊断，因此只能把它视为机制
证据，不能报告为正式安全优势。

同样需要明确：6 个 episode 均未在 40 s 内到达终点。risk-enabled 的两条
无碰撞轨迹完成约 63% 和 69% 路程，说明当前策略偏保守；终点完成率应留给
下一阶段算法设计和新的 held-out 闭环实验解决，不能反向调整本次冻结环境。

## 5. 失败、修订与可追溯性

设计与实现修订全部保留：

1. Amendment 1：原水平路线对部分 seed 的穿越不足；
2. Amendment 2：发现 45 s 生成轨迹与 40 s episode 的计数口径不一致；
3. Amendment 3：改用 registry 明确允许的 smoke seed；
4. Amendment 4：把在线缺失 forecast 从异常终止改为安全停车，并使
   LaserScan 量程覆盖完整场景；
5. Amendment 5：修正世界/里程计坐标系混用的 RMSE 审计，并在已建立
   轨迹后允许单束远距离候选通过预测门控。

Amendment 4 曾完整失败一次，9 个 Gate 中通过 7 个。它的全部 artifact
被保留，没有覆盖。确定性控制重放最大位置误差为 0，证明原大 RMSE
（0.30–0.87 m）来自审计坐标系错误；在在线里程计坐标系中实际 RMSE
为 0.0229–0.0312 m。

## 6. 证据位置

- 最终 Gate：`research_artifacts/mujoco_v3_probabilistic_crossing_smoke_amendment5/gate.json`
- episode 汇总：`research_artifacts/mujoco_v3_probabilistic_crossing_smoke_amendment5/episode_summary.csv`
- paired 效应：`research_artifacts/mujoco_v3_probabilistic_crossing_smoke_amendment5/paired_effects.csv`
- 每步轨迹、配置和 provenance：同目录下 `runs/`
- 失败对照：`research_artifacts/mujoco_v3_probabilistic_crossing_smoke_amendment4/`
- 启动中断记录：`research_artifacts/mujoco_v3_probabilistic_crossing_smoke_amendment3_interrupted_launch/`

## 7. 下一阶段边界

环境和在线概率链路到此冻结。下一阶段应先定义新的算法实验预注册：

- 主指标建议优先采用碰撞率与任务完成率的层级或联合判据；
- 使用未参与本次诊断的 held-out 完整 episode seeds；
- 预先做功效/样本量估计；
- 保留 risk-disabled 与 deterministic-prediction 等消融；
- 不再根据 held-out 结果调整 V3、tracker、Change-Aware IMM 或 Risk V1。

