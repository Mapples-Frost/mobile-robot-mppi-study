# L18：SAC critic 相对 BC 优势诊断预注册

日期：2026-07-15  
状态：在运行 L18 数据前冻结  
研究范围：`u_trap_long_board`、静态障碍、nominal prediction、Memory 关闭、ICODE 关闭

## 1. 为什么做这一轮

L17 已经表明：冻结 BC 后训练出的有界 SAC correction 会改变轨迹，但三个独立训练种子的 fail-closed selector 都保留了 step 0。当前不能区分以下两种解释：

1. critic 能正确识别 correction 的收益，只是 correction 的收益不足；
2. actor 利用了 critic 的估计误差，使 critic 给有害 correction 较高分。

L18 只诊断这一区别，不增加场景复杂度，不联合 ICODE、Memory、OOD 或动态障碍，也不把阈值搜索包装成方法收益。

## 2. 冻结假设

对归一化观测 \(s\)，冻结 BC 动作为 \(a_{\mathrm{BC}}\)，学习后的完整修正动作为 \(a_{\mathrm{corr}}\)。对两个 critic 分别定义

\[
A_i(s)=Q_i(s,a_{\mathrm{corr}})-Q_i(s,a_{\mathrm{BC}}),\qquad i\in\{1,2\}.
\]

保守优势定义为

\[
A_{\min}(s)=\min\{A_1(s),A_2(s)\}.
\]

同时记录 online critics 与 target critics 的 \(A_{\min}\)。本轮的主要问题不是“Q 值是否大”，而是“相对 BC 的排序是否与真实闭环结果方向一致”。

### H1：critic 排序有效

如果 critic 排序有效，则 episode 内平均 \(A_{\min}\) 越高，相同 scene/seed 下 correction 相对 BC 的闭环收益应越好。闭环收益按以下顺序解释：

1. 不损失 BC 原本成功的 episode；
2. 不引入 collision；
3. `return_correction - return_BC` 增大；
4. `goal_distance_BC - goal_distance_correction` 增大。

### H0：critic 排序不可用

若 correction 损失 BC 成功 episode 时仍给出正的保守优势，或优势与配对回报/距离变化没有一致的正向关系，则当前 critic 不足以作为 correction 部署门控。

## 3. 实验单位与配对

- 独立训练重复：L17 v3 的三个 training seeds：`20260721`、`20260722`、`20260723`；
- 候选 checkpoint：每个 training seed 的 `step_000020000.pt`；
- BC 对照：同一 run 的 `initial.pt`，其 deterministic correction 为零；
- 配对任务：同一 scene、同一 episode seed；
- 诊断 seeds：`20280701` 至 `20280715`，属于既有 validation 集，不是 final test；
- 主要独立重复仍是三个 training seeds，45 个 episode 对不能伪称 45 个独立训练重复。

已打开的 L17 final seeds `40301--40320` 不用于阈值选择，也不用于 L18 的确认性结论。若 L18 产生可进入下一阶段的方法，将另行冻结未使用的 final seeds。

## 4. 预先固定的记录字段

每个控制步至少记录：

- online `Q1/Q2` 的 BC 值与 correction 值；
- target `Q1/Q2` 的 BC 值与 correction 值；
- online/target conservative advantage；
- critic disagreement；
- 原始 correction 大小；
- advantage gate alpha；
- 门控后实际 correction 大小。

每个 episode 聚合：

- online/target conservative advantage 的均值、最小值、最大值；
- 正优势步比例；
- gate 接受步比例；
- success、collision、return、final goal distance；
- 相同 seed 下相对 BC 的配对变化。

## 5. 门控只是诊断性消融

实现以下可关闭模式，默认保持 `none`，确保旧 checkpoint 行为不变：

1. `none`：完整 correction，不使用 critic 过滤；
2. `online_hard`：仅当 online \(A_{\min}\ge 0\) 时保留完整 correction；
3. `target_hard`：仅当 target \(A_{\min}\ge 0\) 时保留完整 correction。

零阈值在看数据前固定；本轮不扫描多组阈值。门控作用在 `a_BC -> a_corr` 的 correction 层，而不是把整个 RL prior 退回 GoalWarmStart，因此 `alpha=0` 必须精确恢复冻结 BC。

## 6. 判定规则

只有同时满足以下条件，critic advantage gate 才能进入下一阶段候选：

1. 三个 training seeds 中至少两个呈现优势与配对 `return` 变化同向；
2. BC 原本成功的 episode 不因门控策略而失败；
3. 不增加 collision；
4. 门控不是退化为全接受，也不是退化为全拒绝；
5. 至少一个训练种子出现明确的成功增益，或在成功数不变时产生预先可解释的距离改善。

任一安全非劣条件失败，则不进入新 final test。若优势几乎总为正但闭环结果退化，将其解释为 critic 对 actor correction 的系统性乐观，而不是调整阈值直到得到好看结果。

## 7. 工程验收

- 未启用 advantage gate 时，旧 checkpoint 动作逐元素一致；
- hard gate 拒绝时精确恢复 BC action；
- twin-critic advantage 的 shape、有限值和公式有单元测试；
- metrics、CSV、summary JSON 均记录门控来源和字段；
- 标准 evaluator 与 trainer validation 对相同 checkpoint/seed 仍保持合同一致；
- `scan_guard`、安全仲裁、MuJoCo plant、MPPI cost 和现有 bridge 不修改。

## 8. 本轮不允许声称

- 不声称 critic advantage 是统计置信度；
- 不声称 hard gate 提供稳定性或安全理论保证；
- 不因 validation 上的阈值效果声称泛化；
- 不声称 RL、ICODE、Memory 联合方法已验证；
- 不把 smoke 或 validation 结果写成正式论文主结果。
