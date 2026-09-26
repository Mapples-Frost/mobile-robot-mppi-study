# L17：保守残差策略修正的预注册验证门

> 状态：本节中的假设、数据划分、选择规则和停止规则在查看 L17 训练结果与最终测试结果之前冻结。后续结果只能追加到“结果记录”一节，不得反向修改本节规则。

## 1. 本轮只回答什么问题

L16 已经证明“冻结 BC 策略 + 有界 SAC correction”能够防止整个策略网络被灾难性改坏，但尚未证明 RL correction 带来收益。L17 只检验：

> 在 seen U-trap 中，SAC 是否能在不损失任何配对验证回合成功、不新增碰撞的前提下，对冻结 BC sampling prior 作出可测量的改善？

本轮不同时引入 ICODE、Memory-Augmented MPPI、动态障碍、intrinsic reward、协方差学习或 OOD 域。它们会改变因果问题，因此继续保持关闭。

## 2. 方法与保持不变的链路

控制链仍为：

```text
LaserScan -> local obstacle layer -> MPPI -> safety arbitration -> MuJoCo plant
```

BC base actor、BC normalizer、MPPI 代价、plant、感知和安全链全部冻结。SAC 只输出二维 local-subgoal 参数上的有界 correction：

\[
a(s)=\operatorname{clip}\left(
    a_{\mathrm{BC}}(s)+\Delta a_{\mathrm{SAC}}(s),-1,1
\right).
\]

为避免 reward 鼓励无必要地偏离 BC，actor loss 增加归一化 correction 正则：

\[
\mathcal L_{\mathrm{corr}}=
\frac{1}{B d_a}\sum_{i=1}^{B}\sum_{j=1}^{d_a}
\left(\frac{\Delta a_{ij}}{\Delta a^{\max}_{j}}\right)^2,
\qquad
\mathcal L_{\mathrm{actor}}=
\mathcal L_{\mathrm{SAC}}+0.25\,\mathcal L_{\mathrm{corr}}.
\]

该项约束的是 RL 相对 BC 的修正比例，不是小车控制量本身，也不替代 MPPI cost 或 safety arbitration。

## 3. 实验单位与配对结构

- 独立训练重复：3 个 training seeds，分别为 `20260721`、`20260722`、`20260723`。
- 每个 training seed 使用与其对应的 BC checkpoint，不能交叉使用。
- 固定 validation set：`20280701` 至 `20280715`，用于 checkpoint selection。
- sealed final-test set：`30301` 至 `30320`，只在三个 training seed 的 checkpoint 全部选定后打开一次。
- 同一 training seed 下，不同 checkpoint 必须在完全相同的 scene/seed 行上配对比较。

统计解释上，training seed 才是独立训练重复。每个模型的 15 个 validation episodes 和 20 个 final-test episodes 是嵌套在该训练重复内的重复测量，不能把 `3 x 20` 直接宣称为 60 个独立训练样本。

历史 seeds `101--110` 已被 L16 分析使用，只保留为开发历史，不再承担 L17 最终测试角色。

## 4. Checkpoint 选择规则

step 0 的冻结 BC 行为是每个 training seed 的不可变 reference。每个候选 checkpoint 与 reference 逐场景、逐 seed 对齐，随后执行 fail-closed 规则：

1. reference 成功而候选失败的 episode 数必须为 0；
2. reference 未碰撞而候选碰撞的 episode 数必须为 0；
3. 候选平均终点距离不得高于 reference；
4. 候选还必须至少满足一项“有意义改善”：
   - 至少新增 1 个成功 episode；或
   - 平均终点距离至少降低 `0.005 m`；
5. 通过上述门后，才按成功率、碰撞率、终点距离、return 的字典序更新 best checkpoint。

这意味着更高的平均 return 不能抵消某一个已成功 episode 的失败，也不能抵消新增碰撞。选择器状态、step-zero 行和阈值都进入 checkpoint 与 resume contract，恢复训练时缺失任一部分即拒绝继续。

## 5. 预先冻结的训练设置

配置文件为 `configs/rl/sac_mppi_utrap_conservative_correction_l17.yaml`：

- 训练步数：每个 seed `20,000`；
- actor 开始更新：step `10,001`；
- checkpoint/validation：step `0, 5k, 10k, 15k, 20k`；
- validation episodes：固定 15 个；
- correction 上限与 L16 相同；
- correction penalty weight：`0.25`；
- memory、ICODE、intrinsic exploration、learned covariance：关闭。

`0.25` 只作为本轮预注册系数使用。本轮 final-test 结果不允许用于回调该系数。

## 6. 主要与次要指标

主要指标：

1. success；
2. collision；
3. paired success losses / gains；
4. paired collision regressions。

次要指标：

- final goal distance；
- minimum clearance；
- trajectory length；
- control jerk；
- safety intervention count；
- applied correction magnitude；
- planner compute time（只能在无并行 CPU 竞争的计时实验中比较）。

## 7. Gate 判定与停止规则

L17 seen gate 通过至少要求：

1. 三个 training seeds 均能完成训练、保存、加载和确定性评估；
2. selected checkpoint 在 validation set 上满足上述 fail-closed 规则；
3. sealed final-test pooled collision 不高于对应 BC reference；
4. final test 不出现系统性 success 退化；
5. 至少在成功率或终点距离上出现跨 training seed 可重复的改善趋势，而不是只在单个 seed 偶然改善。

若选择器对三个 training seeds 都保留 step 0，结论是“保守结构有效，但当前 SAC correction 没有证据优于 BC”。若 final test 退化，则 seen gate 失败，停止扩展到 OOD、动态障碍、ICODE 或 Memory，先修正基础 RL 学习问题。

## 8. 执行顺序与信息隔离

1. 先完成代码、单元测试和短 smoke；
2. 对三个 training seeds 独立训练；
3. 仅使用 validation seeds 选择每个 seed 的 checkpoint；
4. 将三个 selected checkpoint 的路径和 global step 写入结果清单；
5. 之后才首次运行 sealed final-test seeds `30301--30320`；
6. final-test 只比较对应 BC step 0 与 L17 selected checkpoint，不据此重新选择模型或调参；
7. 追加结果并如实记录未通过项。

## 9. 结果记录

尚未运行。此处将在模型选择冻结并打开 sealed final-test 后追加，不改写前述预注册内容。

