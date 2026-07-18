# L20：Twin-Critic 共识 LCB 门控预注册

日期：2026-07-15  
状态：冻结于读取 L20 calibration 结果之前  
前置证据：L18 表明 critic advantage 只有弱排序信息；L19 表明固定 raw Q-difference margin 不能跨三个训练种子稳定保护 BC。

## 1. 研究问题

在不重新训练 actor、不加入 ICODE、Memory、OOD 或动态障碍的条件下，尺度不变的 twin-critic 共识分数能否识别可用的 RL correction，并在三个独立训练模型上同时满足 BC non-inferiority？

本轮只检验“现有两个 target critics 的相对一致性是否足以形成部署门控”。若失败，则停止使用同一对 actor critics 构造门控，下一轮转向独立风险估计器或独立 critic ensemble。

## 2. 固定公式

对 frozen BC action (a_{\mathrm{BC}}) 与 correction action (a_{\mathrm{corr}})，定义：

\[
\Delta Q_i(s)=Q_i^{\mathrm{target}}(s,a_{\mathrm{corr}})
-Q_i^{\mathrm{target}}(s,a_{\mathrm{BC}}),\qquad i\in\{1,2\}.
\]

双 critic 平均优势与半分歧为：

\[
\bar A(s)=\frac{\Delta Q_1(s)+\Delta Q_2(s)}{2},
\qquad
U(s)=\frac{|\Delta Q_1(s)-\Delta Q_2(s)|}{2}.
\]

工程共识分数为：

\[
S_\beta(s)=\bar A(s)-\beta U(s).
\]

执行规则为：

\[
a(s)=
\begin{cases}
a_{\mathrm{corr}}, & S_\beta(s)\ge 0,\\
a_{\mathrm{BC}}, & S_\beta(s)<0.
\end{cases}
\]

其中 (eta\ge1) 是无量纲保守系数。当 (eta=1) 时，

\[
S_1(s)=\min(\Delta Q_1,\Delta Q_2),
\]

因此它精确回归 L18 的 target zero-threshold gate。若所有 (Delta Q_i) 同时乘以正常数，门控决策不变；这消除了 L19 的全局 raw-margin 尺度问题，但不保证 critic 本身正确。

“LCB”仅表示工程上的 disagreement-penalized lower-bound score，不是统计置信下界、概率或理论安全保证。

## 3. 候选 beta 与来源

候选值在 L20 数据生成前固定为：

```text
1.0, 1.5, 2.0, 3.0, 5.0, 8.0
```

该网格由已结束的 L18 日志确定。在 L18 ungated 状态上，三个训练模型从 beta 1 到 8 的描述性接受率约从 `0.66--0.84` 降至 `0.14--0.19`，覆盖非退化区间。L18/L19 只用于确定公式和网格，不参与 L20 合格判定。

## 4. 数据隔离

### Calibration

- 独立训练模型：`20260721`、`20260722`、`20260723`；
- 全新 episode seeds：`20281201--20281212`；
- 每个训练模型、每个 beta 与 BC 使用相同 episode seeds；
- 每个训练模型中 BC 只运行一次；
- 用途：选择一个全局 beta，或返回 BC fallback。

### Sealed selection

- 保留仍未打开的 episode seeds：`20281101--20281115`；
- calibration 无合格 beta 时不运行；
- calibration 有合格 beta 时，只比较 BC 与被冻结的单一 beta；
- 不得在 selection 结果上重新选择 beta。

## 5. 实验单位和控制变量

- 独立重复单位：3 个 independently trained checkpoints；
- episode seeds 是模型内部的配对重复，不当作 36 个独立训练重复；
- 固定静态 `u_trap_long_board`、nominal prediction、Memory off、ICODE off；
- plant、sensor、initial state 与 MPPI RNG 按 episode seed 完整配对；
- 并行只缩短运行时间，不比较 wall-clock planner time。

## 6. Calibration 合格条件

一个 beta 必须同时满足：

1. pooled paired BC-success losses = 0；
2. collision regressions = 0；
3. 每个 training seed 的 mean paired return delta 均不小于 0；
4. 每个 training seed 的 gate acceptance 位于 `[0.05, 0.95]`；
5. pooled 至少 1 个 success gain，或 mean goal-distance improvement 不小于 `0.005 m`。

若多个 beta 合格，按以下字典序选最大者：

1. pooled success gains；
2. 最差 training-seed mean return delta；
3. pooled mean return delta；
4. pooled mean goal-distance improvement；
5. beta，较大者优先。

若无 beta 合格：

```text
selected_mode = bc_fallback
selected_beta = null
selection_seeds_opened = false
```

## 7. Selection 合格条件

只有 calibration 合格才检查：

1. 45 个 paired episodes 中无 BC-success loss；
2. 无 collision regression；
3. 每个 training seed mean return delta 不小于 0；
4. 每个 training seed gate 非退化；
5. pooled 至少 1 个 success gain，或 `0.005 m` mean distance improvement。

任一失败均记录为 calibration overfit，不进入后续 OOD、动态障碍或 ICODE+RL 联合实验。

## 8. 工程与数据质量 Gate

- 未开启 `lcb` 时必须保持 L17--L19 行为；
- beta=1 必须数值等价于 conservative minimum gate；
- 对 (Delta Q_1,Delta Q_2) 同乘正常数时，决策必须不变；
- beta 网格、condition mapping、training seeds、episode seeds 不一致时 fail closed；
- CSV 复合键无重复，关键字段无缺失，数值无 NaN/Inf；
- BC raw correction 精确为零；
- 输出 config snapshot、checkpoint provenance、episode/paired/step CSV 和 selection JSON。

## 9. 停止条件与允许结论

若无 beta 合格，不继续增加 `10、12、20` 等 beta，也不把 selection seeds 用于调参。允许结论仅为：当前同一对 actor critics 的 disagreement penalty 仍不足以形成安全门控。

不允许把本轮称为理论置信界，不允许声称跨场景、跨物理域泛化，也不允许声称 RL、ICODE 或联合系统已验证。
