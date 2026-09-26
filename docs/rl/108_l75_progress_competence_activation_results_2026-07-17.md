# L75：基于在线进度的 RL 激活门控结果

日期：2026-07-17  
性质：预注册开发实验；未使用封存种子；不能作为独立确认结果

## 1. 结论

L75 的预注册开发 Gate **未通过**。这一结论不是由碰撞或程序错误导致，而是方法
假设本身被数据否定：把“最近两秒目标距离下降不足”直接解释为“此时应增加 RL
sampling prior 权重”，会形成闭环正反馈。

机器人可能因为狭窄通道中的合理减速而暂时进度不足。L75 在这种状态下提高 RL
权重；如果 RL 又进一步降低进度，下一时刻的停滞分数会更高，门控会继续保持或
增加 RL 权重。该机制没有提供“本次 RL 修正是否真的有效”的撤回路径。

因此：

- L75 规则被淘汰；
- 不打开 L75 封存种子；
- 不将本轮失败包装成“接近通过”；
- 保留复杂度门控、target twin-critic LCB、ICODE 与安全链；
- 下一轮只门控 SAC 对冻结 BC 的**增量修正**，不再关闭冻结 BC 脚手架。

## 2. 实验完整性

- 独立训练模型块：3；
- 场景：`clean_single_obstacle`、`narrow_corridor`、
  `u_trap_long_board`；
- 每个模型块、每个场景使用 5 个新开发 seed；
- 条件：冻结 BC、始终采用复杂度外门控的 SAC-LCB、进度激活 SAC-LCB；
- 总 episode：`3 × 3 × 5 × 3 = 135`；
- 期望 episode：135，实际 episode：135；
- 重复主键：0；
- 碰撞回归：0；
- L74 及更早受保护 seed：0；
- L75 封存 seed：0；
- 数值有限性检查：通过。

控制 step 是 episode 内的重复技术测量，不作为额外独立样本。三个模型块才是训练
随机性的独立重复。

## 3. 主要结果

进度激活 SAC-LCB 相对冻结 BC：

| 范围 | success gain | success loss | 净变化 | 平均最终距离改善 |
|---|---:|---:|---:|---:|
| 全部 45 对 | 2 | 16 | -14 | -0.603 m |
| 模型块 0 | 0 | 4 | -4 | -0.597 m |
| 模型块 1 | 1 | 8 | -7 | -0.988 m |
| 模型块 2 | 1 | 4 | -3 | -0.224 m |

按场景拆分：

| 场景 | 冻结 BC 成功 | 进度门控成功 | success gain/loss | 平均最终距离改善 |
|---|---:|---:|---:|---:|
| single obstacle | 14/15 | 14/15 | 1 / 1 | +0.000 m |
| narrow corridor | 9/15 | 1/15 | 1 / 9 | -1.784 m |
| U-trap | 15/15 | 9/15 | 0 / 6 | -0.025 m |

进度门控平均 RL 外层权重为 `0.372`，反而高于始终采用复杂度门控条件的
`0.344`。这不是代码违反了逐步乘法约束，而是闭环轨迹发生分叉：进度门控较早
撤掉了有用的 BC/SAC 轨迹引导，进入更困难状态后，停滞信号和几何复杂度长期同时
升高。

作为对照，始终采用复杂度门控的 SAC-LCB 相对冻结 BC 在 45 对中为
`4 gain / 4 loss`，净成功变化为 0，平均最终距离改善 `+0.176 m`。按场景为：

- single obstacle：净 `-1`；
- narrow corridor：净 `+2`，平均最终距离改善 `+0.613 m`；
- U-trap：净 `-1`。

这说明训练后修正并非完全无效；它在窄通道中有可重复价值，但目前还不能稳定判断
何时应该撤回。

## 4. 失效机理

L75 错误地把两件事合并成了一个外层权重：

1. 冻结 BC 给出的轨迹族脚手架；
2. SAC 在冻结 BC 基础上的增量修正。

当进度历史尚未准备好或检测到正常前进时，L75 把整个 learned prior 的外层权重
降为零，此时回退的是 `GoalWarmStartPrior`，而不是冻结 BC。因此它可能在进入
窄通道之前就丢失了必要的轨迹族引导；等到真正停滞时再把 SAC 放大，已经太晚。

轨迹审计还显示：

- SAC-LCB 成功的 narrow-corridor episode 也会出现短暂低进度；
- 失败 episode 的区别主要是低进度状态长期持续，而不是第一次出现低进度；
- 失败 episode 的平均 observation-support/OOD 分数高于成功 episode；
- target critic LCB 仍会在部分高 OOD 状态接受修正，说明 critic 的正值不能单独
  被解释为已校准置信度。

## 5. 下一轮的受控修改

下一轮不再使用“停滞时打开全部 RL prior”。改为：

\[
a_{\mathrm{deploy}}
=
a_{\mathrm{BC}}
+
c_{\mathrm{support}}(o)
\,
g_{\mathrm{LCB}}(o,a)
\,
\Delta a_{\mathrm{SAC}},
\]

其中：

- 冻结 BC 始终保留；
- `g_LCB` 继续用 target twin-critic 对增量修正做符号筛选；
- `c_support` 只根据训练 observation normalizer 的支持分数连续衰减 SAC 增量；
- 外层 LaserScan complexity gate、near-goal fallback、ICODE、MPPI 和安全仲裁不变。

这里的 `c_support` 是工程支持度，不是概率，也不声称是校准置信区间。该修改直接
针对 L75 暴露的问题：在不熟悉状态中退回冻结 BC，而不是把整个 learned prior
撤掉或在停滞后持续放大修正。

## 6. 产物

- 配置：`configs/rl/progress_competence_activation_l75.yaml`
- 预注册：`docs/rl/107_l75_progress_competence_activation_prereg_2026-07-17.md`
- 汇总：
  `results/research_platform/rl/l75_progress_competence_activation_20260717_v1/summary/l75_summary.json`
- episode 配对：
  `results/research_platform/rl/l75_progress_competence_activation_20260717_v1/summary/l75_paired_effects.csv`
- 三个模型块的逐步轨迹和 checkpoint 哈希均保存在同一结果目录。
