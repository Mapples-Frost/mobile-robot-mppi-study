# L86 锚定协方差 RL 开发结果

日期：2026-07-17  
证据级别：单训练块开发结果，不是论文确认性结论

## 1. 实现

L86 将强固定协方差 `speed=(1.75,0.75)` 作为安全锚点，SAC 只输出每维不超过 `log(1.25)` 的有界对数残差。actor 零输出和 gate 零权限均精确退化到锚点。ICODE、MPPI、MuJoCo、观测、安全仲裁与对偶精度约束保持不变。

新增实现具有默认关闭兼容性；未配置 anchor 的历史 covariance-only checkpoint 仍使用原映射。

## 2. 训练结果

训练目录：

`results/research_platform/rl/l86_anchored_covariance_seed20268821_v1/`

20,000-step 单块训练：

| 指标 | step-zero 锚点 | 选中 checkpoint（20k） |
|---|---:|---:|
| 验证 cross-track RMSE | 0.035368 m | 0.035210 m |
| mean return | 74.473932 | 74.488810 |
| success rate | 100% | 100% |
| collision rate | 0% | 0% |
| covariance scale mean | (1.750, 0.749) | (1.683, 0.713) |
| 最终对偶乘子 | 0 | 0.187 |

训练内 checkpoint 同时满足精度约束和更高 return，说明策略产生了非零上下文残差，且没有出现 L85 的大幅精度劣化。

## 3. 新 seed 配对评估

结果目录：

`results/research_platform/rl/l86_anchored_covariance_development_eval_20260717_v1/`

设计：1 个独立训练块 × 4 场景 × 5 个全新物理 seed × 3 条件，共 60 个闭环回合。所有条件 100% 成功、0 碰撞。

### 相对强固定锚点

| 指标（RL - fixed speed） | 均值差 | 95% bootstrap CI | 门槛 |
|---|---:|---:|---|
| cross-track RMSE | -0.000151 m | [-0.001132, +0.000857] m | 上界 <= +0.002 m：通过 |
| time to goal | -0.020 s | [-0.130, +0.090] s | 上界 < 0：失败 |
| control jerk | -0.002404 | [-0.003148, -0.001630] | 改善 |
| trajectory length | -0.002028 m | [-0.007196, +0.002711] m | 不确定 |
| planner compute mean | +3.093 ms | [+1.187, +5.277] ms | 明确开销 |

预注册主门槛为精度非劣、时间优越和安全非劣同时成立。L86 的精度与安全门槛通过，时间门槛失败，因此 `primary_gate_passed=false`。

### 相对默认协方差

L86 的 RMSE 低 0.013546 m，control jerk 低 0.020195，但到达时间反而高 1.110 s。该结果再次说明默认 `(1,1)`、强固定 `(1.75,0.75)` 和学习策略位于不同 Pareto 位置，不能只用单一指标宣称全面优越。

## 4. 与 L85 的机制对照

- L85 自由策略：相对强固定锚点快 1.755 s，但 RMSE 高 3.899 mm；时间通过、精度失败。
- L86 锚定残差：RMSE 均值反而低 0.151 mm，且 jerk 更低，但时间只快 0.020 s、CI 跨零；精度通过、时间失败。

因此锚定参数化确实解决了精度退化，却也把策略限制在强固定解附近，当前场景中没有显著的到达时间改进空间。

## 5. 研究判断

本轮不否定 RL+ICODE 大方向，但否定以下两个过强表述：

1. “任何 covariance RL 都能稳定优于调好的固定 MPPI”；
2. “加入精度约束后即可同时获得 L85 的全部速度收益”。

当前最合理的下一步不是继续调 SAC，而是先验证任务是否真的存在**上下文相关、相互冲突的最优采样分布**。若所有场景都由 `(1.75,0.75)` 近似统一支配，RL 没有足够的可学习优势，继续训练只会得到近似恒等映射。

## 6. 下一 Gate：采样异质性诊断

下一阶段先固定 ICODE 与 MPPI，在场景 × 物理域 × 动态障碍条件上运行 covariance oracle grid，回答：

- 不同上下文的最优 scale 是否显著不同；
- context oracle 相对 best global fixed 的闭环收益是否超过统计噪声和 RL 推理开销；
- 哪些可观测量能够预测最优 scale；
- unseen 域是否仍保持可预测异质性。

只有 context oracle 在闭环上同时给出明确收益上界，才继续训练 preference/context-conditioned RL；否则论文应把 RL 贡献转向数据采集、计算预算分配或困难场景触发，而不是强行学习协方差。

## 7. 验证

- anchored covariance / precision / checkpoint / paired gate 定向测试：76 passed；
- 全仓测试：577 passed；
- L85 与 L86 summary 已使用精度非劣 + 时间优越 + 安全非劣的显式门槛重新生成；
- 未执行 git add、commit 或 push。
