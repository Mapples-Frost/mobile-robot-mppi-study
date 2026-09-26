# L92 动态协方差安全独立确认：未通过，停止场景标签支线

日期：2026-07-18  
结论等级：预注册独立 context-block confirmation；主 Gate 未通过。

## 1. 完整性

- 6 个 L91 之前已存在的动态运动几何；
- 5 个冻结协方差候选；
- 4 个 selection seeds、6 个 held-out evaluation seeds；
- 300/300 个 MuJoCo 闭环 episode 完成；
- 冻结 ICODE、MPPI、LaserScan、scan_guard 与安全仲裁；
- 无 scene-label planner 输入、无障碍物真值输入、无残差再训练。

selection 映射使用 4 种候选，6 个 context 中有 4 个不同于最强全局固定候选
`turn=[0.75,1.75]`，因此“不同场景会在开发集上选出不同协方差”的描述性异质性仍存在。

## 2. Held-out 结果

context oracle 相对最强全局固定 `turn` 的 36 个 evaluation 配对：

| 指标 | 配对均值差 | 分层 bootstrap 95% CI |
|---|---:|---:|
| Success | +0.0556 | [-0.1389, +0.2500] |
| Collision | 0.0000 | [-0.1389, +0.1667] |
| Elapsed time | -2.350 s | [-7.914, +2.720] s |
| Final goal distance | -0.0725 m | [-0.4358, +0.3205] m |
| Control jerk | -0.00304 | [-0.01759, +0.01251] |
| Minimum clearance | +0.0318 m | [-0.0656, +0.1312] m |

Success 区间下界没有大于 0，collision 区间上界也大于 0，因此两项预注册安全条件均
失败，`primary_gate_passed=false`。

## 3. 与 L91 的关系

L91 在 4 个场景上得到 +0.30 success、-0.15 collision 的强信号；L92 使用新的 6 个
运动几何后，只剩 +0.0556/0.0000，所有主要区间跨零。这说明 L91 的增益不能泛化为
“知道动态场景类别即可稳定选择更安全协方差”。

因此不能基于 L91 训练 scene-class policy，也不能把 L91 单独作为论文主正向结果。

## 4. 科研决策

停止“按动态场景标签选择协方差”支线。保留 L89 已独立通过的路线几何 contextual
bandit；动态障碍改为部署边界与 fallback 问题：RL 负责它已证明有效的路线级探索，实时
LaserScan/temporal risk 只决定何时退回更保守的传统采样，而不预测场景类别。

下一步先在同一条路线的 clean/dynamic-crossing 条件下做固定候选 headroom gate。只有
clean 偏好路线 RL 动作、dynamic crossing 偏好保守动作且 held-out 配对稳定，才实现
LaserScan 风险门控。

原始结果：
`results/research_platform/rl/l92_dynamic_covariance_safety_confirmation_20260718_v1/summary.json`。
