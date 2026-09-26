# L19：跨训练种子的 critic advantage margin 校准预注册

日期：2026-07-15  
状态：运行 L19 calibration 数据前冻结  
前置证据：L18 表明非负 twin-critic advantage 含有弱排序信息，但零阈值门控仍损失 BC-success episodes。

## 1. 研究问题

L19 检验一个比 L18 更窄的问题：**在完全独立的新 calibration episodes 上，能否为三个独立训练模型选择同一个保守 target-critic advantage margin，使 correction 保留可测收益，同时不损失 BC？**

这不是重新训练 RL，也不是引入新方法模块。ICODE、Memory、OOD、动态障碍和 RL actor 更新继续关闭。

## 2. 固定定义

对 target twin critics：

\[
A_{\mathrm{target}}(s)=
\min_{i\in\{1,2\}}
\left[
Q^{\mathrm{target}}_i(s,a_{\mathrm{corr}})
-Q^{\mathrm{target}}_i(s,a_{\mathrm{BC}})
\right].
\]

给定 margin \(m\)，执行规则为：

\[
a(s)=
\begin{cases}
a_{\mathrm{corr}}, & A_{\mathrm{target}}(s)\ge m,\\
a_{\mathrm{BC}}, & A_{\mathrm{target}}(s)<m.
\end{cases}
\]

候选 margin 在查看 L19 数据前固定为：

```text
0.000, 0.005, 0.010, 0.020, 0.040, 0.080
```

该网格由 L18 的优势量级确定，因此 L19 是在新数据上的校准研究；不能把 L18 与 L19 calibration 合并后宣称完全前瞻的确认性结论。

## 3. 数据隔离

### Calibration

- training seeds：`20260721`、`20260722`、`20260723`；
- episode seeds：`20281001--20281012`；
- 每个 training seed 对所有 margin 和 BC 使用相同 episode seeds；
- 用途：只允许选择一个全局 margin，或选择 BC fallback。

### Selection

- 预留 episode seeds：`20281101--20281115`；
- 在 calibration 选择冻结前不得运行；
- 若 calibration 无合格 margin，不打开 selection seeds；
- selection 只运行 BC 与冻结的单一 margin，不重新搜索阈值。

L17 final seeds `40301--40320`、L18 seeds `20280701--20280715` 均不参与 L19 选择。

## 4. 实验单位

- 独立训练重复：3 个 independently trained checkpoints；
- episode seeds 是每个训练模型内部的配对重复，不得当成独立训练重复；
- scene/seed/plant/sensor/MPPI RNG 全部配对；
- 并行运行只缩短耗时，不比较 planner wall-clock。

## 5. Calibration 合格条件

一个 margin 只有同时满足以下条件才合格：

1. 三个 training seeds 合并后 `BC success losses = 0`；
2. `collision regressions = 0`；
3. 每个 training seed 的 mean paired return delta 均不小于 `0.0`；
4. 每个 training seed 的 mean gate acceptance 位于 `[0.05, 0.95]`；
5. pooled 至少获得 1 个 success gain，或 mean goal-distance improvement 不小于 `0.005 m`。

这些是 fail-closed 条件；不能用平均成功率抵消某个 paired BC-success loss。

## 6. 多个合格 margin 的固定选择顺序

若多个 margin 合格，按以下字典序选择最大者：

1. pooled success gains；
2. 三个 training seeds 中最差的 mean return delta；
3. pooled mean return delta；
4. pooled mean goal-distance improvement；
5. margin 数值本身，较大者优先。

若无 margin 合格：

```text
selected_mode = bc_fallback
selected_margin = null
```

并停止 L19，不运行 selection seeds。

## 7. Selection 通过条件

只有 calibration 选出 margin 后才检查：

1. 全部 45 个 paired episodes 无 BC-success loss；
2. 无 collision regression；
3. 每个 training seed mean return delta 不小于 `0.0`；
4. gate 在每个 training seed 上保持非退化；
5. pooled 至少 1 个 success gain，或 `0.005 m` mean distance improvement。

任一条件失败，则该 margin 只记录为 calibration overfit，不进入新 final test。

## 8. 工程验收

- margin runner 对 BC 只运行一次，不随网格重复计算；
- condition、threshold、training seed、episode seed 构成唯一实验键；
- selector 输入、阈值网格、排序 tuple、选择原因写入 JSON；
- 候选网格不完整、训练 seed 重复、episode seed 漂移或 CSV 重复时 fail closed；
- BC raw correction 必须精确为零；
- 所有关键数值必须有限；
- 未开启 margin 时保持 L18/L17 行为回归。

## 9. 本轮不允许声称

- 不把 Q-margin 称为统计置信区间；
- 不声称校准 margin 跨场景或跨物理域泛化；
- 不在 calibration 与 selection 之间改候选网格、规则或 checkpoint；
- 不在同一 selection 集上二次选阈值；
- 不声称 RL、ICODE、Memory 联合体系已验证。
