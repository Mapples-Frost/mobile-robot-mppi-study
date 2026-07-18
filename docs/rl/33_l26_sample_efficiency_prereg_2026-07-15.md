# L26：场景复杂度门控 RL–MPPI 的采样效率预注册

日期：2026-07-15  
性质：开发集确认性实验；运行前冻结。L25 与 L26 的封存测试种子均不打开。

## 1. 研究问题

L25 在固定 `K=200` 时证明了局部 LaserScan 几何门控能够在空旷区域精确退回 traditional MPPI，并在三个局部阻塞场景中显著改善成功率。但固定一个 `K` 无法判断：RL prior 是真正提高了有限采样预算下的样本效率，还是只在当前某个采样规模下偶然有效。

L26 检验：

1. Complexity-Gated RL 在 `K=50/100` 时是否改善局部阻塞场景；
2. 低预算 gated 方法是否能够达到高预算 traditional MPPI 的成功率；
3. 随着 `K` 增大，traditional MPPI 是否追平，或 learned prior 是否仍有持续优势；
4. 空旷区域是否在所有 `K` 下逐控制步精确回退；
5. 性能提升是否伴随碰撞或不可接受的计算代价。

## 2. 冻结因素与实验单位

- 方法：`traditional_mppi`、`complexity_lcb`；
- MPPI 采样数：`K=50,100,200,400`；
- 空旷对照：`clean_dynamics`；
- 局部阻塞层：`clean_single_obstacle`、`narrow_corridor`、`u_trap_long_board`；
- 独立训练重复：checkpoint seeds `20260721/22/23`；
- 每个 checkpoint 使用十个新的开发 episode seeds：`20281801–20281810`；
- 每个场景–方法–K 单元共 30 个 episode，嵌套于三个独立 checkpoint；
- 总 episode 数：`3 × 10 × 4 × 2 × 4 = 960`。

checkpoint 是策略训练层面的独立重复。相同 checkpoint 下的 episode、控制步和不同 K 都是配对或嵌套测量，不得伪称为独立训练模型。

## 3. 控制变量

- 冻结 L25 的 actor、twin target critics、`beta=2` LCB 和 complexity 映射；
- nominal prediction、LaserScan、local obstacle layer、scan guard 和 safety arbitration 不变；
- memory 与 ICODE 关闭，避免把动力学学习混入 sampling-prior 实验；
- 同一 checkpoint–scene–episode seed–K 下两种方法配对；
- 每个 checkpoint 内使用固定随机种子打乱完整运行顺序；
- 不修改 L25 数据、标签或阈值。

## 4. 主要终点

1. 每个场景、方法和 K 的成功率；
2. gated 相对 traditional 的配对成功增加/损失；
3. 最终目标距离；
4. 碰撞、安全仲裁、最小净空和控制平滑性；
5. 平均与最大 planner 时间；
6. gate alpha、active fraction 与空旷场景逐步回退误差。

## 5. 开发 Gate

全部条件在数据生成前固定：

1. 960 个 episode 完整、无重复、无非有限指标；
2. L25/L26 封存种子使用数均为零；
3. 空旷场景在所有 K 下 `v`、`omega`、goal distance、collision 与 safety override 逐步完全一致，且 gate alpha 恒为零；
4. gated 相对 traditional 的碰撞回归为零；
5. `K=50/100` 的三个阻塞场景合计净成功增益至少 18；
6. `complexity_lcb@K=100` 在至少两个阻塞场景内，与 `traditional@K=400` 的成功率差不低于 `−0.05`；
7. `complexity_lcb@K=100` 的阻塞场景平均 planner 时间不超过 `traditional@K=400` 的 60%。

Gate 失败时不改阈值、不删除不利场景，并继续封存所有测试种子。Gate 通过也只说明开发证据允许进入一次性确认实验，不等同于论文最终结论。

## 6. 可能的解释分支

- 低 K 有优势且高 K 逐渐收敛：支持“RL prior 提高有限预算下样本效率”；
- 所有 K 均保持优势：支持“方向性 prior 持续改变搜索质量”，但不能只讲低预算；
- 仅单一 K 有优势：认为结果不稳定，不支持采样效率机制；
- 低 K 无优势或出现安全回归：L26 失败，冻结测试集并回到机制诊断。

## 7. 封存边界

- L25 sealed seeds：`20281711–20281730`；
- L26 sealed seeds：`20281811–20281830`；
- 本轮仅允许 `20281801–20281810`；
- 未满足开发 Gate 前，不得使用 `--allow-sealed-test`。
