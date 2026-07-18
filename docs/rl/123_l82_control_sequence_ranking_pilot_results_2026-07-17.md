# L82：控制序列排序保真度 mechanism pilot 结果

日期：2026-07-17  
结论：**不升级 formal；不采用 ICODE elite sequence distillation；转入 covariance-only RL sampling 诊断**

## 1. 完整性

本轮在 L82 预注册后执行：

- 3 个冻结的 ICODE/参数量匹配 MLP 模型块；
- 4 条高动态路径：`accel_turn`、`chicane`、`hairpin_unseen`、`reverse_s_unseen`；
- 每条路径固定控制步 `20,50,80,110`；
- 每个 anchor 64 条、每条 36 步的候选控制序列；
- nominal、MLP、ICODE 使用完全相同的候选张量；
- true cost 来自恢复同一 MuJoCo snapshot 后实际执行完整候选序列；
- memory、RL 和 learned gate 均关闭；场景无障碍，不混入感知误差；
- 未发生碰撞、NaN、Inf、缺失 anchor 或人工删除样本。

总计 48 个物理 anchor、3,072 条真实 36 步反事实轨迹，以及 144 条
`anchor × prediction model` 排序记录。MuJoCo snapshot 单元测试覆盖 command delay、PI
积分状态和 torque history，重复分支在 `atol=1e-12` 下复现。

工件目录：

`results/research_platform/rl/l82_ranking_pilot_20260717_v1/`

## 2. ICODE 相对 MLP 的配对结果

正值对 Spearman/elite recall 表示 ICODE 更好；负值对 regret/JS divergence 表示 ICODE 更好。

| 指标 | ICODE - MLP | 期望方向 | pilot 结果 |
|---|---:|---:|---|
| Spearman rank correlation | `-0.006953` | `> 0` | 不支持 |
| top-10% elite recall | `-0.005952` | `>= 0` | 不支持 |
| normalized regret | `+0.013876` | `<= 0` | 不支持 |
| MPPI weight JS divergence | `+0.003580` | `<= 0` | 不支持 |

三个模型块的 Spearman difference 分别为：

| block | Spearman diff | elite diff | regret diff | weight-JS diff |
|---:|---:|---:|---:|---:|
| 0 | `-0.010763` | `-0.008929` | `+0.001216` | `+0.007608` |
| 1 | `-0.003073` | `-0.017857` | `+0.011160` | `+0.001696` |
| 2 | `-0.007023` | `+0.008929` | `+0.029251` | `+0.001436` |

即 0/3 block 的平均 Spearman difference 为正。unseen 路径也没有反转：

- unseen Spearman difference：`-0.013555`；
- unseen elite recall difference：`-0.017857`；
- unseen normalized regret difference：`+0.026941`。

按场景看，seen 的 `accel_turn` 和 `chicane` 接近持平；负差异主要扩大于
`hairpin_unseen` 和 `reverse_s_unseen`。这不是由单个坏 checkpoint 造成，因为三个模型块方向一致。

## 3. 被否定的窄假设

本轮不否定 L59/L62/L65/L68 已经确认的 ICODE 闭环优势。它否定的是以下更窄的机制解释：

> ICODE 闭环优于 MLP，是因为它对固定 36 步候选控制序列的真实累计成本排序更准确，因此可直接把 ICODE elite 序列作为 RL proposal teacher。

pilot 中该解释没有证据支持。根据预注册，不能在看到结果后调整 anchor、候选噪声或只保留
seen 场景再宣称通过。因此不支付完整 formal 的额外计算预算，也不把 ICODE-generated elite
当作下一阶段监督标签。

## 4. 与既有闭环结果并不矛盾

L82 是“固定一个当前 target、固定一批 36 步 open-loop sequence”的排序实验；真实闭环 MPPI
每 0.1 s 会重新观测、更新 polyline target、重新采样并只执行第一步。ICODE 的结构优势可能体现为：

- 连续重规划时局部 (v,\omega) 响应更稳定；
- control-affine 结构在动作变化时保持更有用的局部梯度/相对误差；
- 即使完整序列 top-1 排序不优，适当采样覆盖和每步重新加权仍能产生更好的首动作；
- MLP 的平均预测误差和固定序列排序较好，但误差方向在 receding-horizon feedback 下更不利。

这些只是后续可检验解释，不作为本轮已证明结论。

## 5. 方法转向

RL + ICODE 大方向保持不变，RL 仍作为 MPPI sampling prior 的组成部分，但学习对象从“动作均值修正”改为：

> **保持 conventional/previous-sequence mean 不变，只学习状态相关的采样协方差尺度。**

即：

\[
q_\phi(U\mid o_t)
=
\mathcal N\!\left(
\mu_{\mathrm{MPPI}}(o_t),
\operatorname{diag}\left[(s_\phi(o_t)\odot\sigma_0)^2\right]
\right).
\]

这仍属于 RL-guided MPPI sampling prior，但不再让不稳定的 RL mean correction 覆盖已经较强的
traditional prior。简单状态可保持 (s\approx 1)，复杂/OOD 状态可在约束范围内增加或收缩
速度/转向搜索。

在训练前必须先做 covariance-scale oracle diagnostic：若不同状态下最佳尺度没有变化，或 oracle
选择相对固定 baseline 没有收益，则 covariance-only RL 也应停止，而不是盲目训练。

## 6. 解释边界

- 本轮是 mechanism pilot，不是正式论文统计结果；
- 结果不能证明 MLP 总体优于 ICODE，因为既有 sealed 闭环证据相反；
- 未测试障碍感知、动态障碍或实车；
- 未训练新的 RL，不声称 covariance-only 策略有效；
- 不把未通过的 direction Gate 改写成正结论。
