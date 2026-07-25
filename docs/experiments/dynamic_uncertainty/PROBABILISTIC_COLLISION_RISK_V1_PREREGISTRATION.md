# Probabilistic Collision Risk V1 资格预注册

日期：2026-07-23  
平台：Windows native  
阶段：冻结预测器之后、闭环 MPPI 之前  
状态：在实现和查看资格结果前冻结

## 1. 研究问题

给定机器人候选轨迹和 Change-Aware IMM 输出的未来二维混合高斯分布，能否得到一个
因果、有限、可解释、适合批量 MPPI 计算的碰撞风险量？

本阶段不评价闭环避障成功率，不修改 V3 障碍物或预测器，也不使用 held-out 或 sealed
seed。所有资格检查只使用解析构造和预先定义的合成几何。

## 2. 数据契约

每个动态障碍预测包含：

- 预测起始时间；
- 固定预测步长；
- `H × M × 2` 个位置均值；
- `H × M × 2 × 2` 个位置协方差；
- `H × M` 个模式概率；
- 障碍物物理半径。

模式概率必须非负且每个时刻和为 1。协方差必须有限、对称、半正定。风险接口不得
接收未来真值状态、真实模式或真实 change flag。

## 3. 冻结数学定义

令机器人候选位置为 \(r_t\)，某一高斯分量的障碍位置为
\(X_{tm}\sim\mathcal{N}(\mu_{tm},\Sigma_{tm})\)。组合半径为

\[
R = r_\mathrm{robot} + r_\mathrm{obstacle} + r_\mathrm{margin}.
\]

当 \(d=\|\mu-r\|>R\) 时，取从机器人指向障碍均值的单位向量
\(n=(\mu-r)/d\)，径向标准差为

\[
\sigma_n=\sqrt{n^\top\Sigma n+\sigma_\min^2}.
\]

碰撞圆盘包含于其朝向均值的切平面事件中，因此使用

\[
\bar p_{tm}=\Phi\left(\frac{R-d}{\sigma_n}\right)
\]

作为该分量单步碰撞概率的保守上界。当分量均值已经位于组合半径内时，冻结为
\(\bar p_{tm}=1\)，使规划器 fail closed。

混合高斯单步上界为

\[
\bar p_t=\sum_m w_{tm}\bar p_{tm}.
\]

时间相关性未知，因此不使用独立性假设。整条候选轨迹同时输出：

- 未裁剪概率质量：\(C_\mathrm{risk}=\sum_t\bar p_t\)，用于连续 MPPI 代价；
- union bound：\(\bar p_\mathrm{any}=\min(1,\sum_t\bar p_t)\)，用于解释与审计；
- 最大单步风险：\(\max_t\bar p_t\)；
- 是否超过冻结的硬风险阈值。

## 4. MPPI 接口

风险端口默认关闭。关闭时，历史 MPPI 的代价必须逐元素完全相同。打开时：

\[
J \leftarrow J
  + w_\mathrm{risk}C_\mathrm{risk}
  + I(\max_t\bar p_t\ge p_\mathrm{hard})J_\mathrm{hard}.
\]

未来分布通过 `RobotObservation.auxiliary["probabilistic_obstacle_forecasts"]`
提供。端口打开但预测缺失时必须显式报错，不允许静默退化为“无动态风险”。

本阶段只验证接口和排序。`weight`、硬阈值及硬惩罚是后续 development smoke 的控制
参数，不在本阶段用闭环结果定稿。

## 5. 冻结资格 Gate

必须同时满足：

1. 非法形状、权重、半径、时间步或协方差被拒绝；
2. 所有合法输出有限且属于 `[0,1]`；
3. 分量均值位于组合半径内时风险为 1；
4. 均值在安全区外时，距离更近风险严格更高；
5. 其他量相同时，径向协方差更大风险严格更高；
6. 混合风险对模式权重线性，且模式置换不改变结果；
7. 穿越候选轨迹的风险严格高于绕行候选；
8. union bound 不小于任意单步风险；
9. MPPI 端口关闭时与历史代价逐元素相同；
10. MPPI 端口打开时穿越候选得到更高代价；
11. 预定义二维高斯 Monte Carlo 检查中，上界违反数为 0；
12. Windows CPU 上 `K=600, H=36, M=4` 的参考批量风险计算不超过 25 ms。

## 6. 解释限制

- 该量是圆盘碰撞概率的保守切平面上界，不是任意二维混合高斯圆积分的精确值。
- 时间 union bound 不假设各步独立，因此可能偏保守。
- 感知中的目标检测与关联误差尚未包含；后续必须单独实现和验证在线轨迹关联。
- Predictor held-out 结果显示模式切换期间仍偏自信，因此保留 0.10 m 显式安全裕量。
- 本 Gate 通过不等于闭环安全通过。

