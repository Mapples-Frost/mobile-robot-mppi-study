# L215 复杂障碍导航开发与封存协议

日期：2026-07-19  
性质：结果产生前冻结的 development protocol；不是正式结论。

## 1. 研究问题

L214 的 clean point-goal benchmark 只隔离了动力学失配与耦合机制，不能回答复杂障碍导航问题。L215 回答：

> 在静态复杂几何和 MuJoCo 物理失配下，Value-Consistent ICODE、Residual-Conditioned RL guidance、Reliability-Adaptive HSS 与 MPPI 的完整耦合，能否相对简单组合提高安全到达率，并保持等 rollout 预算？

核心机制冻结为：

1. ICODE control-affine residual 用于 MPPI rollout dynamics；
2. RL Actor 生成 MPPI 候选控制序列，不直接越过 MPPI 输出控制；
3. value-aligned ICODE 提供控制任务相关的残差预测；
4. ICODE ensemble reliability 调节 Actor 候选比例与 terminal value 权限；
5. scan_guard、安全仲裁和 LaserScan → local obstacle layer 链路始终开启；
6. memory 关闭，避免与主要消融混杂。

## 2. 已知失败与本轮起点

L213 在 `lab_complex`、180 control steps 下四臂均为 0/15 成功。该结果必须保留。L215 的首次可解性审计得到：

- Traditional MPPI 在 360 步内停在局部障碍前；
- 当前四个 Actor-guided 变体可绕过障碍，行驶约 7.3–7.6 m；
- 360 步结束时距离目标约 0.49–0.59 m，0 collision；
- Full Proposed 最后 25 步仍持续接近目标，因此 360 步包含 execution-window 截断。

所以 L215 允许把所有方法的共同执行上限提高到 600 步，并把 time-to-goal 作为效率指标。不得只给 Proposed 延长时长。

## 3. 数据隔离

### 3.1 Development seeds

只允许在 seeds `48,49,50` 上：

- 审计场景可解性；
- 修复可恢复执行、指标/provenance 和明显工程缺陷；
- 调整不改变核心机制的训练稳定性、统一 MPPI 预算、episode 上限和近目标权限切换；
- 选择唯一候选配置。

### 3.2 Sealed seeds

封存 seeds 在 development Gate 通过前不得运行、查看或用于阈值选择。候选配置冻结后，封存集只运行一次。若封存结果失败，不得回调并重新声称同一批 seeds 是独立确认。

## 4. Development 场景与物理域

复杂场景：

1. `lab_complex`：多箱体和多圆柱组合；
2. `narrow_corridor`：连续窄通道与局部阻塞；
3. `u_trap_long_board`：需要非贪心绕行的 U 型几何。

物理域：

- `nominal_seen`：标称 MuJoCo plant；
- `combined_unseen`：组合未见动力学失配。

障碍物只通过合成 LaserScan、scan_guard 和 local obstacle layer 进入 planner，不允许把全局真值障碍物直接注入 MPPI。

## 5. 七方法矩阵

1. Traditional MPPI；
2. ICODE-MPPI；
3. RL-driven MPPI；
4. Simple combination：ordinary ICODE + fixed RL guidance/value；
5. Value-only：value-aligned ICODE + fixed guidance；
6. HSS-only：ordinary ICODE + reliability-adaptive guidance；
7. Full Proposed：value-aligned ICODE + reliability-adaptive guidance/value。

所有方法使用相同的每拍 100 次 model rollout 预算；RL 变体使用两轮、每轮 50 条候选。不得把 timestep 当作独立统计样本。

## 6. Development Gate

只有同时满足以下条件才允许冻结并开启 sealed benchmark：

1. 所有轨迹、metrics、resolved config 和 provenance 完整；
2. 至少两个复杂场景中存在安全成功轨迹，证明场景和执行窗口可解；
3. Full Proposed 在 development blocks 上相对 Simple combination 的成功率方向有利；
4. Full Proposed 的平均 final distance 方向有利；
5. Full Proposed 不增加碰撞；
6. 等 rollout 预算校验通过；
7. Value-only 与 HSS-only 至少有一个模块在成功率或连续距离上呈现独立正向作用；
8. 任何修复均有单元/回归测试，且不得关闭安全链。

Development Gate 只决定是否有资格做确认实验，不构成论文证据。

## 7. 封存统计

确认实验以 seed 为独立单位，scene 和 physics domain 为 seed 内重复分层。报告：

- success rate、collision rate；
- final goal distance、time-to-goal、trajectory length；
- minimum clearance、stuck/spin steps；
- mean absolute omega、control jerk；
- planner mean/max/p95 compute time；
- reliability authority、guided fraction 与 terminal-value authority。

主要比较为 Full Proposed vs Simple combination。使用 seed-cluster paired bootstrap，10,000 次重采样；同时报告七方法描述统计和 `2×2` value-alignment × adaptive-HSS 主效应/交互项。

## 8. 允许与禁止的解释

允许：开发结果用于定位失败、选择单一冻结候选。  
禁止：筛选好看的 seed、删除失败 episode、把 development 当确认集、把 0 collision 等同于全局安全保证，或把公开 control-affine residual 实现声称为原始 ICODE 全部理论保证。

## 9. Development amendment：定位隔离与终点交接

本修订在 sealed benchmark 开启前完成，只依据 development seed 48 的失败审计；它不改变 ICODE、RL guidance、value alignment 或 reliability-adaptive HSS 的核心机制。

1. 主复杂导航 benchmark 使用带既有微小传感噪声的 MuJoCo ground-truth pose/twist，隔离本文研究的动力学失配与规划变量。原始 wheel-odometry 模式保留为独立 localization stress test，不混入主因果比较。
2. 七种方法共同启用同一近目标控制律：仅在距目标 0.80 m 内限制平移速度，并在航向偏差较大时允许 `v=0, omega!=0` 原地对准。目标区外不改变 Actor 或 MPPI 的障碍绕行能力。
3. Adaptive-HSS arms 在 0.80 m 至 0.40 m 内连续将 Actor proposal 与 incremental learned terminal value 交还给 baseline MPPI；0.40 m 内 learned authority 为零。该修复补足“只减少 Actor 候选数、但 Gaussian mean 仍由 Actor 初始化”的实现语义漏洞。
4. 旧的 wheel-odometry 负向结果与修订后的 qualification 结果必须同时保留。只有完整 development matrix 通过第 6 节 Gate，才允许另行预注册从未运行的新 sealed seeds。
