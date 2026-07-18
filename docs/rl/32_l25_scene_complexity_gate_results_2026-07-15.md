# L25：基于 LaserScan 场景复杂度的 RL–MPPI 门控结果

日期：2026-07-15  
性质：开发集筛选实验，不是独立测试集结论，也不是论文最终结果。

## 1. 结论

L25 给出了目前最清晰的机制证据：门控器能够在无障碍局部几何中将 RL 贡献精确降为零，并在三个存在阻塞几何的场景中显著改善传统 MPPI 的成功率。全部 480 个 episode 均未碰撞。

但是，预注册的开发 Gate **未通过**。失败项来自事先把 `clean_single_obstacle` 归入“simple”后，对整个 simple 组施加了过低的平均门控强度上限；该场景在当前 `K=200` 下并不简单，传统 MPPI 为 `0/30`，门控策略恰好在其中启动并救回 `20/30`。本轮不允许事后修改标签或阈值，因此独立测试种子继续封存。

这支持“根据局部可观测几何按需使用 RL”的研究方向，但尚不能证明该方法在所有场景中优于 always-on RL 或 frozen-BC prior。

## 2. 实验设计

- 三个独立训练 checkpoint：训练种子 `20260721`、`20260722`、`20260723`；
- 每个 checkpoint 使用 10 个相同的开发 episode 种子；
- 四个场景、四种方法，组成严格配对比较；
- MPPI 采样数固定为 `K=200`；
- 总计 `3 × 10 × 4 × 4 = 480` 个 episode、`124334` 个控制 step；
- 方法：traditional MPPI、frozen-BC prior、always-on LCB、complexity-gated LCB；
- 门控输入只来自 LaserScan，不访问 MuJoCo 障碍物真值；
- memory 默认关闭；预测、MPPI、scan guard 和安全仲裁链保持不变。

所有 episode 的执行顺序由固定调度种子随机化。episode 种子在同一场景、checkpoint 和方法间配对；三个 checkpoint 是独立训练重复，episode 不被错误地当成 30 个独立训练模型。

## 3. 成功率与安全

| 场景 | Traditional MPPI | Frozen BC | Always LCB | Complexity LCB |
|---|---:|---:|---:|---:|
| clean dynamics | 30/30 | 10/30 | 0/30 | **30/30** |
| single obstacle | 0/30 | 0/30 | 0/30 | **20/30** |
| narrow corridor | 0/30 | **29/30** | 28/30 | 24/30 |
| U-trap | 0/30 | **28/30** | 21/30 | 26/30 |

所有 480 个 episode 的碰撞数均为 0。这个结果说明门控方法相对 traditional MPPI 在三个阻塞场景中分别产生 `20`、`24`、`26` 个成功增益且没有成功损失；但 narrow corridor 中 gated 低于 frozen BC 和 always LCB，U-trap 中低于 frozen BC，因此不能声称统一占优。

按训练 checkpoint 查看，complexity-gated LCB 的成功数为：

| 场景 | seed 20260721 | seed 20260722 | seed 20260723 |
|---|---:|---:|---:|
| clean dynamics | 10/10 | 10/10 | 10/10 |
| single obstacle | 6/10 | 7/10 | 7/10 |
| narrow corridor | 9/10 | 7/10 | 8/10 |
| U-trap | 8/10 | 9/10 | 9/10 |

因此主要趋势不是由某一个训练 checkpoint 单独造成的。

## 4. 门控是否真的按场景工作

场景复杂度定义为三个 LaserScan 分量的最大值：正前方近障、双侧夹窄和近障光束密度。分数经过预注册的软/硬阈值映射为 RL 混合系数 `alpha`。

| 场景 | 平均复杂度分数 | 平均 alpha | active fraction | traditional fallback fraction |
|---|---:|---:|---:|---:|
| clean dynamics | 0.000 | 0.000 | 0.000 | 1.000 |
| single obstacle | 0.449 | 0.375 | 0.601 | 0.397 |
| narrow corridor | 0.540 | 0.460 | 0.754 | 0.230 |
| U-trap | 0.508 | 0.406 | 0.682 | 0.262 |

相对 always-on LCB，门控后的整体平均 `alpha` 降低 `60.65%`。这说明 RL 不是一直接管采样 prior，而是随局部几何变化开启或回退。

在无障碍 `clean_dynamics` 中，还对 traditional 与 gated 的 `30` 个 checkpoint–episode 配对、`2286` 个控制 step 做了逐步精确核对。以下量的最大绝对差均为 `0`：

- 执行线速度；
- 执行角速度；
- 到目标距离；
- collision 标记；
- safety override 标记。

这不是“平均接近”，而是当前随机种子和数值实现下的逐步精确 fallback。

## 5. 与 always-on RL 和 frozen BC 的关系

Complexity LCB 相对 always LCB 的成败配对变化为：

- clean dynamics：`+30 / -0`；
- single obstacle：`+20 / -0`；
- narrow corridor：`+2 / -6`；
- U-trap：`+8 / -3`。

相对 frozen BC：

- single obstacle：`+20 / -0`；
- narrow corridor：`+1 / -6`；
- U-trap：`+2 / -4`。

因此门控解决了“简单、无障碍环境中 RL 反而伤害传统 MPPI”的关键问题，也能在 single-obstacle 场景救回 always-on 方法；但当前线性复杂度映射会在部分窄通道时刻过早回退，尚未实现对强 prior 的统一超越。

## 6. 计算开销

| 方法 | 平均 planner 时间 | 最大 planner 时间 |
|---|---:|---:|
| traditional MPPI | 4.82 ms | 25.13 ms |
| frozen BC | 7.22 ms | 47.42 ms |
| always LCB | 7.22 ms | 30.07 ms |
| complexity LCB | 7.33 ms | 46.98 ms |

本轮没有 planner deadline miss。门控增加的是轻量 LaserScan 统计，主要额外开销仍来自神经 prior 与 critic 推理。

## 7. 预注册 Gate 判定

通过的条件包括：

- 复杂场景平均门控强度与 active fraction 位于预设区间；
- 相对 always-on 的 alpha reduction 达标；
- 复杂场景相对 traditional MPPI 有净成功增益和距离改善；
- simple 组没有成功率退化；
- 没有碰撞回归。

失败的条件：

- 预注册 simple 组平均 `alpha = 0.287 > 0.15`；
- 预注册 simple 组 active fraction `= 0.459 > 0.25`。

失败不是因为安全性差或控制性能退化，而是因为“全局障碍物数量少”等价于“RL 不应启用”这一场景标签假设不成立。单个障碍物也可能构成决定性的局部阻塞。为避免事后选择，本轮结论保持为 Gate fail，封存测试种子不打开。

## 8. 数据质量与可复现性

- 期望 episode 行数：480；实际：480；
- 重复主键：0；
- episode 指标有限值检查：通过；
- gate step 数值有限值检查：通过；
- 所有方法与场景覆盖：通过；
- 独立测试种子使用数：0；
- checkpoint、配置、Git SHA、随机调度和每条轨迹均已记录；
- 无障碍场景的 clearance 缺失是“没有障碍物”的结构性缺失，不是日志损坏。

主要产物：

- `results/research_platform/rl/l25_scene_complexity_development_multiseed_20260715_v1/audit.json`
- `results/research_platform/rl/l25_scene_complexity_development_multiseed_20260715_v1/development_gate.json`
- `results/research_platform/rl/l25_scene_complexity_development_multiseed_20260715_v1/summary.json`
- `results/research_platform/rl/l25_scene_complexity_development_multiseed_20260715_v1/eda.md`

## 9. 下一步（L26）

冻结 L25 结果，不用本轮数据继续调阈值。下一轮使用新的开发种子，并将 MPPI 采样预算设为 `K = 50, 100, 200, 400`，检验以下可证伪问题：

1. RL prior 的主要价值是否表现为低采样预算下的样本效率提升；
2. traditional MPPI 是否随着 K 增大逐步追平；
3. complexity gate 是否能在无障碍区域保持精确 fallback，同时在局部阻塞出现时启用 RL；
4. 门控和采样预算是否存在交互效应，而不是只报告某一个 K 的偶然结果。

只有 L26 开发 Gate 通过，才允许一次性打开已封存的独立测试种子。
