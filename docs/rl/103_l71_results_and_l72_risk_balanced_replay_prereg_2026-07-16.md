# L71 结果与 L72 场景—结果二维平衡 Replay 预注册（2026-07-16）

## L71 结论

L71 完整运行了 189 个闭环 episode，artifact matrix、checkpoint hash、冻结 BC
精确回退和 seed 隔离均通过。按预注册规则，只有 block 2 的 step 30,000
checkpoint 合格：

- 9 个配对 episode 中 success gains = 2，success losses = 0；
- collision regressions = 0；
- mean final-distance improvement = 0.3377 m；
- mean correction LCB acceptance = 0.3207。

block 0 没有候选满足 non-inferiority；block 1 的 BC 已经 9/9 成功，所有候选均
没有达到预注册的有效改善阈值。因此只有 1/3 block 选择非零 SAC，L71 Gate
失败，sealed confirmation seeds 保持未使用。

L71 预注册将 episode return 作为成功数与距离都相同时的最后 tie-break，但标准
`ExperimentRunner` summary 不输出训练环境 return。汇总器最初将缺失字段显示为
0；现已改为显式 `null` 和 `return_tiebreak_available=false`。本次 Gate 失败不依赖
该 tie-break：block 0/1 没有合格候选，而 block 2 的两个合格候选在相同净成功
增益下由预注册中更靠前的距离改善项明确区分（0.3377 m 对 0.0112 m）。

## 失败机制

L70 使用 scene-balanced replay，三个场景的 transition 数量得到平衡，但同一场景
内部仍可能由成功 episode 主导。训练产物中的完整 episode outcome counts 表明，
成功与失败轨迹比例在不同训练 seeds 间差异明显。SAC actor 直接优化的 twin critics
因此对有害 correction 的估计不稳定；L71 中 block 0 的 LCB 平均放行约 29%–52%，
但所有候选的平均终点距离均变差。

这与早期 L18–L24 的结论一致：继续扫描 critic 阈值不是可靠修复。L72 不改变
LCB 阈值，而从 critic 的训练数据分布入手。

## L72 单因素改动

新增 `scene_outcome_balanced` replay：

1. minibatch 先在所有已出现的场景之间等量分配；
2. 对每个场景，若已经同时存在完成的成功与失败 episode，则按固定 50:50 抽取；
3. 只有一个已完成 outcome 类别时，在该场景内普通采样，不制造标签；
4. 未完成 episode 不参与已有成功/失败两类时的平衡抽样；
5. 环形 replay overwrite 仍由 transition id 防止错误重标注。

相对 L70，以下因素全部冻结：三个场景、BC bases、三组 ICODE、30k steps、reward、
actor/critic 网络、学习率、correction scale、MPPI K=100、MuJoCo plant、传感器、
scan_guard、memory=false 和 raw-policy checkpoint selector。L72 只更换 replay
sampler，并使用新的 SAC seeds `20260751–20260753` 和新的训练内 validation seeds。

## 资格门槛

L72 继续沿用 L70 的严格门槛：

- 三个训练均完整达到 30,000 steps；
- 三个场景均进入 replay；
- 每个训练产物记录 outcome counts 和 minibatch scene/success fractions；
- raw-policy selector 不允许 success loss、collision regression 或平均距离退化；
- 至少 2/3 独立训练 blocks 选择非零 checkpoint。

若失败，不开启新的 deployment selection 或 sealed test。若通过，再为 L72 单独预
注册 deployment-aligned checkpoint selection，不能复用 L71 development seeds。

## 运行中断与原样重试记录（2026-07-17）

首次正式运行目录后缀为 `v1`。Codex 任务被外部中断时，三个训练进程同时终止；
checkpoint 审计显示三者均为：

- `global_step = 0`；
- `episodes = 0`；
- 只完成预注册的 step-0 validation；
- 没有 `step_000005000.pt` 或 `latest.pt`；
- 没有任何 SAC/environment training transition 可用于分析。

因此 `v1` 保留为中断 artifact，不删除、不 resume、不纳入任何效果统计。正式重试
写入全新的 `v2` 目录，继续使用完全相同的配置、BC/ICODE 配对、训练 seeds 和
validation seeds。该重试没有依据 step-0 结果修改方法、阈值或样本量，属于操作中断
后的原样重复，而不是自适应调参。
