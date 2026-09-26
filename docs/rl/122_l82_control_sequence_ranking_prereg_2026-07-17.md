# L82：控制序列排序保真度诊断预注册

日期：2026-07-17  
性质：development mechanism diagnostic；在任何 L82 MuJoCo 候选分支数据产生前冻结

## 1. 研究问题

L60--L62 观察到：参数量匹配 MLP 的 offline H36 RMSE 优于 ICODE，但 ICODE 的闭环路径跟踪优于 MLP。L82 检验一种可证伪解释：

> ICODE 不一定使所有状态预测的均方误差最小，但可能更准确地保持控制候选的成本次序，因此 MPPI 更容易选到真实低成本序列。

这不是新的 ICODE 性能确认，也不直接训练 RL。它是选择后续 RL 学习对象的机制实验。

## 2. 固定比较

对同一 MuJoCo plant snapshot、同一目标、同一批候选控制序列 (\{U_k\}_{k=1}^{K})，分别计算：

\[
\hat S_k^{\mathrm{nom}},\qquad
\hat S_k^{\mathrm{MLP}},\qquad
\hat S_k^{\mathrm{ICODE}},\qquad
S_k^{\mathrm{true}}.
\]

前三者使用各自 prediction dynamics rollout；`true` 通过从完全相同的 MuJoCo snapshot 恢复并执行整个候选序列获得。所有轨迹使用同一 MPPI cost contract 评分。

冻结项：

- MuJoCo model、plant 参数和 controller cost；
- L57 ICODE checkpoints 与 L60 参数量匹配 MLP checkpoints；
- horizon、control bounds、candidate set 和随机种子；
- memory disabled；
- 无 RL prior、无 gate、无 safety override；
- 首轮只使用无障碍高动态 path-tracking 场景，以隔离动力学排序，不混入 perception 和 obstacle representation。

“无 safety override”仅用于受 control bounds 限制的离线反事实分支；正式闭环控制链不变。

## 3. 候选构造

每个 anchor state：

1. 由冻结的 conventional MPPI/reference controller 给出中心序列；
2. 使用固定高斯噪声构造相同的 (K) 条候选；
3. 按 action bounds clip；
4. 第 0 条候选固定为中心序列；
5. 三种预测模型和 true plant 必须使用逐元素完全相同的候选张量。

smoke 固定：`K=32`，block 0、`accel_turn` 的控制步 `5,15`。  
跨块 mechanism pilot 固定：`K=64`，控制步 `20,50,80,110`，场景为
`accel_turn`、`chicane`、`hairpin_unseen` 和 `reverse_s_unseen`。  
formal development 固定：`K=128`，每场景 8 个 anchor，控制步为
`10,30,50,70,90,110,130,150`。

smoke 与 mechanism pilot 只判断工件、效应方向和是否值得支付 formal
计算预算，不用于通过第 7 节的方向 Gate。pilot 在三个冻结模型块上运行，且不得依据
pilot 结果改变 formal 的候选数、anchor 或六个场景。

anchor 在参考闭环 episode 的预注册控制步上采集，不依据当时结果人工挑选。候选序列不是独立实验重复，只是同一 anchor 内的成对评价样本。

## 4. 主要指标

每个 anchor、每个预测模型计算：

1. Spearman rank correlation：
   \[
   \rho_s = \operatorname{corr}(\operatorname{rank}(\hat S),
   \operatorname{rank}(S^{\mathrm{true}})).
   \]
2. true top-10% elite recall：预测 top-10% 与 true top-10% 的交集比例；
3. predicted argmin true regret：
   \[
   R =
   \frac{S^{\mathrm{true}}_{\arg\min_k\hat S_k}-
   \min_k S^{\mathrm{true}}_k}
   {\operatorname{IQR}(S^{\mathrm{true}})+10^{-9}};
   \]
4. MPPI weight Jensen--Shannon divergence；
5. top-1/top-5 命中和所有成本/轨迹有限性。

主要比较按优先级为：

1. ICODE vs parameter-matched MLP；
2. ICODE vs nominal；
3. MLP vs nominal。

## 5. 实验单位与模型块

独立工程模型块沿用既有冻结 checkpoint 对：

| block | ICODE | MLP |
|---:|---|---|
| 0 | L57 seed `20261201` | L60 seed `20261201` |
| 1 | L57 seed `20261202` | L60 seed `20261202` |
| 2 | L57 seed `20261203` | L60 seed `20261203` |

统计描述单位是完整的 `model block × scene`；同一 scene 内多个 anchor 是重复测量，不冒充独立样本。首轮 formal development 场景：

- `accel_straight`；
- `accel_turn`；
- `chicane`；
- `sweep`；
- `hairpin_unseen`；
- `reverse_s_unseen`。

## 6. 完整性 Gate

进入结果解释前必须满足：

1. 每个 snapshot 恢复后，相同控制分支重复两次的状态轨迹逐元素接近；
2. 候选 tensor hash 在 nominal/MLP/ICODE/true 之间一致；
3. 所有成本有限，候选数、horizon、state/action dimension 完整；
4. 没有碰撞或超出预注册 control bounds；
5. model/checkpoint/config/git SHA/seed/snapshot time 全部写入审计文件；
6. 不允许删除不利 anchor；
7. smoke 只验证工件，不用于支持研究主张。

## 7. 方向 Gate

L82 不设置只追求显著性的单阈值，而使用预注册的分叉规则：

### 分叉 A：进入 ICODE-guided proposal RL

若满足全部条件：

- ICODE vs MLP 的 mean paired Spearman difference 为正；
- 至少 2/3 model blocks 的 paired difference 为正；
- ICODE 的 elite recall 不低于 MLP；
- ICODE 的 normalized regret 不高于 MLP；
- unseen paths 的方向与总体一致。

则允许把 ICODE elite/weighted sequence 作为 proposal policy 的初始化 teacher。

### 分叉 B：进入局部门控/自适应预算

如果总体不满足 A，但 ICODE 优势集中在可提前识别的状态组（例如高加速度、高转率或高模型 disagreement），则后续优先学习受限 authority 或 adaptive sampling allocation。

### 分叉 C：进入 active residual collection

如果 ICODE 在排序指标上不优于 MLP/nominal，或 snapshot 机制无法可靠构造反事实，则不使用 ICODE 作为 proposal teacher。RL 转向主动采集对 ICODE 最有信息量的轨迹，并以相同真实交互预算下的 unseen-domain 改善为目标。

## 8. 解释边界

- L82 只研究有限候选集上的成本排序，不证明全局最优性；
- true cost 是冻结 MuJoCo 平台的反事实结果，不等于实车真值；
- Spearman/elite recall 改善不自动等于最终闭环成功率改善；
- 若进入 A，仍必须通过独立的 RL proposal 闭环消融；
- 不从 development anchor 外推正式论文统计结论。
