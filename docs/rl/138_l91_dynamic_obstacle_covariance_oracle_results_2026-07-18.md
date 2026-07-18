# L91 动态障碍采样协方差 Oracle：安全信号显著，原主 Gate 未通过

日期：2026-07-18  
结论等级：预注册 development oracle；非可部署策略结果。

## 1. 完整性

L91 完成 160/160 个 MuJoCo 闭环 episode：1 个 clean 场景、3 个动态障碍强度、5 个
固定采样协方差、3 个 selection seeds 和 5 个独立 evaluation seeds。所有条件共享冻结的
ICODE checkpoint、MPPI、LaserScan、local obstacle layer、scan_guard 和最终安全仲裁。

oracle 只在 episode 开始前以场景名称选择候选；planner 仍只从 LaserScan 获得障碍物，
没有读取 MuJoCo 障碍轨迹真值。场景名称不是实车可用输入，所以本结果只衡量 headroom。

## 2. Selection 得到的映射

最强全局固定候选是 `turn=[0.75,1.75]`。按场景选择得到：

| 场景 | 选择 |
|---|---|
| clean | `baseline=[1.0,1.0]` |
| dynamic easy | `speed=[1.75,0.75]` |
| dynamic moderate | `broad=[1.75,1.75]` |
| dynamic hard | `speed=[1.75,0.75]` |

四个场景全部不同于最强全局固定候选，并使用了三种不同动作，说明动态障碍条件下存在比
L90 局部路径切换更强的采样异质性。

## 3. Held-out evaluation

场景 oracle 相对最强全局固定候选，20 个配对 episode 的结果为：

- success rate：`+0.30`，95% CI `[+0.05, +0.55]`；
- collision rate：`-0.15`，95% CI `[-0.40, 0.00]`；
- elapsed time：`-3.01 s`，95% CI `[-9.025, +3.165] s`；
- final goal distance：`-0.541 m`，95% CI `[-1.079, -0.121] m`；
- control jerk：`-0.0206`，95% CI `[-0.0368, -0.00745]`；
- minimum clearance：`+0.0498 m`，95% CI `[-0.0259, +0.1424] m`。

方向上，oracle 同时提高成功、减少碰撞、缩短平均用时、减小终点距离并降低 jerk。尤其
final-distance 和 jerk 的区间不跨零，说明差异并非只来自单个失败 episode。

## 4. 为什么仍判定原主 Gate 失败

L91 预注册要求 elapsed-time 的分层 bootstrap 95% CI 上界严格小于 0。本轮上界为
`+3.165 s`，因此 `time_gate_passed=false`，完整主 Gate 必须如实判为失败。不能在看到
success 与 collision 的强信号后，事后把主指标改成安全指标并宣称 L91 通过。

但这也不是“没有结果”。原设计只有 4 个场景层级，时间区间主要受场景间差异支配；同时
selection 本来就先按 success/collision 排序，时间只在安全并列时比较。因此 L91 暴露了
一个值得独立确认、且比耗时更重要的安全假设：动态场景自适应采样能否稳定提高成功率并
降低碰撞率。

## 5. 决策

本轮不直接训练部署策略。下一轮使用 L91 之前已经存在的 6 个不同动态运动几何作为新
context block，使用全新 selection/evaluation seeds，预注册二元安全结局的分层区间。
只有该独立安全确认通过，才进入 LaserScan/temporal-scan 可观测策略。

原始结果：
`results/research_platform/rl/l91_dynamic_covariance_context_oracle_20260718_v1/summary.json`。
