# L27：零复杂度门控的行为保持推理 fast-path 预注册

日期：2026-07-15  
性质：L26 计算开销失败后的独立开发实验；运行前冻结。

## 1. 动机

L26 证明 Gated RL `K=100` 的成功率显著超过 Traditional `K=400`，但 planner 时间比例为 0.945，未达到预注册的 0.60。代码审计发现：当前实现即使已经由 LaserScan complexity 确定外层 gate alpha 为零，仍会先执行 actor、target twin critics 和 decoder，再把 learned mean 的贡献乘零。

L27 不改算法输出，只检验一个工程优化：

> 当 complexity gate、near-goal gate 或 safety fallback 已经确定最终 alpha 为零，且策略不学习 covariance 时，直接返回同一个 conventional prior，跳过 learned inference。

## 2. 设计

- 条件：原始 `standard_complexity` 与 `zero_complexity_fastpath`；
- 固定 `K=100`；
- 场景与 L26 相同：clean、single obstacle、narrow corridor、U-trap；
- checkpoint：`20260721/22/23`；
- 新开发 episode seeds：`20281901–20281910`；
- 每个场景–条件单元 30 episode；总计 240 episode；
- checkpoint–scene–episode 内配对，共用 seed；
- 运行顺序在每个 checkpoint block 内随机化；
- L25/L26/L27 封存种子全部禁用。

## 3. 首要要求：行为完全相同

fast-path 只有在未优化实现最终也会得到 alpha=0 时才能触发。开发 Gate 要求：

1. 两条件逐 step 的 executed `v/omega` 和 goal distance 最大绝对差为零；
2. collision 与 safety override 不一致数为零；
3. episode success 和 collision 不一致数为零；
4. active gate 区域仍运行 actor/critics，不允许近似推理或改阈值；
5. `learn_covariance=true` 时禁止 fast-path。

任何行为差异都使 L27 失败，不允许用“误差很小”放宽。

## 4. 计算 Gate

在行为完全一致的前提下：

- clean 场景 learned-inference skip fraction ≥0.99；
- 三个 blocking 场景合并 skip fraction ≥0.10；
- clean 平均 planner 时间降低 ≥15%；
- blocking 合并平均 planner 时间降低 ≥5%。

这些阈值只评价同一 `K=100` 下优化前后差异，不用于事后改写 L26 的失败 Gate。

## 5. 解释边界

- 通过：说明可以用门控逻辑避免无效神经推理，且保持控制输出不变；
- 时间未通过但行为通过：保留优化，但不能声称有实用加速；
- 行为不通过：立即禁用 fast-path；
- L27 无论是否通过，都不自动打开任何 sealed test seeds。
