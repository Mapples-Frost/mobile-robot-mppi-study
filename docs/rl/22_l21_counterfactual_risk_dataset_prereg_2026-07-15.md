# L21：独立风险估计的反事实分支数据预注册

日期：2026-07-15  
状态：冻结于生成 L21 数据之前  
前置证据：L20 beta 2 在 36 个 calibration episodes 中救回 5 次、损失 1 次，说明现有 critic 共识包含有用信号，但同一对 actor critics 不能提供零回归风险判断。

## 1. 本轮目标

L21 不立即训练或部署新的门控。先构造能够回答以下问题的数据：

> 在完全相同的 MuJoCo、LaserScan、MPPI 随机状态下，仅把当前一步 frozen-BC latent action 替换为 SAC correction，之后两条分支都恢复 frozen BC，这一次局部干预会让未来短时结果变好还是变坏？

该数据用于后续训练与 actor critic 解耦的风险估计器。若反事实重放不确定、有效分支过少或标签几乎单一，则停止训练，不用普通 episode 标签替代。

## 2. 为什么不用整条 episode 的成败给每一步贴标签

若一条 300-step 轨迹失败，不能推断 300 个动作都危险。把同一失败标签复制到每一步会造成错误 credit assignment，并把同一 episode 的数百行误当作独立样本。

L21 采用单步干预：

1. 从 frozen BC reference trajectory 的同一状态开始；
2. baseline branch 当前步执行 (a_{\mathrm{BC}})；
3. candidate branch 当前步执行 L20 beta-2 已接受的 (a_{\mathrm{corr}})；
4. 从下一步起，两条分支都使用各自观测下的 frozen BC 闭环策略；
5. 使用相同 seed 和相同 MPPI random-number stream；
6. 比较未来 20 个控制步。

干预变量只有当前一步的 latent correction；后续均恢复同一个 BC policy class。

## 3. 可复现性前置 Gate

在写入任何 L21 样本前，必须验证同 seed 重置并重放相同 BC latent actions 时：

```text
max action error = 0
max encoded-observation error = 0
max reward error = 0
max goal-distance error = 0
termination flags identical
```

当前开发前置检查在 seed `20281301` 上连续 80 步全部为零误差。正式 collector 仍会在每个 branch replay 时检查 reference observation；误差超过 `1e-10` 立即 fail closed。

## 4. 固定干预策略

- reference/follow-up policy：candidate checkpoint 内冻结的 BC actor；
- correction proposal：同一 checkpoint 的 deterministic SAC correction；
- pre-gate：L20 target twin-critic consensus LCB，`beta=2`；
- 只有 beta-2 接受的 correction 才形成干预样本；
- beta-2 拒绝的状态只计入 coverage，不伪造“安全 correction”标签；
- MPPI、plant、sensors、perception、local obstacle layer 和 scan guard 全部保持运行。

## 5. Branch schedule 与数据划分

每个 episode 固定候选 branch steps：

```text
10, 30, 50, 70, 90, 110
```

branch horizon：20 control steps。

### Development train

```text
episode seeds = 20281301--20281308
```

### Development validation

```text
episode seeds = 20281309--20281310
```

### Sealed model test

```text
episode seeds = 20281311--20281315
```

本轮只允许运行 train/validation。test seeds 在风险模型架构、loss、阈值和 checkpoint selection 冻结前不得运行。L20 未打开的 `20281101--20281115` 继续保留为未来闭环 selection，不用于监督模型训练。

三个 training checkpoints `20260721/22/23` 使用相同 episode split 和 branch schedule。独立统计重复仍是 training checkpoint；branch steps 嵌套于 episode，不按独立重复计数。

## 6. 分支结果与标签

定义：

\[
\Delta R_H=R_H^{\mathrm{corr}}-R_H^{\mathrm{BC}},
\]

\[
\Delta d_H=d_H^{\mathrm{BC}}-d_H^{\mathrm{corr}}.
\]

其中正的 (Delta d_H) 表示 correction 分支更接近目标。

优先级最高的离散事件：

- `collision_regression`：BC 无碰撞、correction 碰撞；
- `success_loss`：BC 在 horizon 内成功、correction 未成功；
- `success_gain`：BC 未成功、correction 成功。

预注册三分类标签：

```text
harmful:
  collision_regression
  OR success_loss
  OR (return_delta <= -0.5 AND distance_improvement <= -0.03 m)

beneficial:
  not collision_regression
  AND not success_loss
  AND (success_gain
       OR (return_delta >= +0.5 AND distance_improvement >= +0.03 m))

neutral:
  otherwise
```

连续的 return/distance targets 始终保存，离散标签只是后续建模候选，不能替代原始测量。

## 7. 输入特征与禁止泄漏字段

允许风险模型使用的当前时刻特征：

- normalized observation；
- frozen BC latent action；
- candidate latent action 与 correction；
- online/target paired-Q advantages；
- critic advantage mean、half-disagreement、LCB score；
- correction magnitude；
- branch step。

标签和未来结果严禁进入输入：future reward、future distance、success/collision outcome、episode return delta。训练/验证拆分必须按 `(training_seed, episode_seed)` group 完成，不能随机拆控制步或 branch rows。

## 8. 数据格式

每个 run 必须输出：

- `samples.npz`：训练数组；
- `samples.csv`：可人工检查的展平表；
- `metadata.json`：配置、seed split、checkpoint、git SHA、维度、label contract；
- `reference_episodes.csv`：reference 长度、成功、碰撞与 beta-2 coverage；
- `config_snapshot.json`。

每行至少包含 training seed、episode seed、branch step、split、feature vector、两个 branch 的全部摘要、连续 target 和三分类 label。

## 9. 数据充分性 Gate

只有同时满足以下条件，下一阶段才允许训练风险模型：

1. 所有 replay observation error 不大于 `1e-10`；
2. 无重复 `(training_seed, episode_seed, branch_step)`；
3. 所有数值有限，数组维度跨 checkpoint 一致；
4. train 至少 60 个有效 accepted branches；
5. validation 至少 12 个有效 accepted branches；
6. train 中 `harmful` 与 `beneficial` 各至少 8 个；
7. validation 中 `harmful` 与 `beneficial` 各至少 2 个；
8. 三个 training checkpoints 均贡献 train 和 validation 样本。

若标签不足，下一步应增加新的 episode seeds、物理扰动或干预覆盖；不得通过复制少数类 branch rows、随机拆分同一 episode 或打开 sealed test 来补数量。

## 10. 当前不允许声称

- 不声称 branch 20-step label 等价于完整 episode 成败；
- 不声称 supervised risk estimator 已训练或有效；
- 不把嵌套 branch rows 当作独立实验重复；
- 不声称 simulator counterfactual 可以直接部署到实车；
- 不开启 ICODE、Memory、OOD 或动态障碍；
- 不修改 scan_guard 的最高安全优先级。
