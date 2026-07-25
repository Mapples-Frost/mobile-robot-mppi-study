# 单动态障碍论文实验：Baseline 与消融预注册草案

状态：**DRAFT FOR REVIEW / 未获准运行**

日期：2026-07-25

适用范围：单动态障碍 MuJoCo 闭环实验

## 0. 本草案纠正的问题

上一版确认实验把旧 Actor 检查点 `source_actor` 设为主要对照。该对照只能回答“新旧神经网络权重谁更好”，不能回答论文需要的“完整方法是否优于标准 MPPI”。

本方案作如下纠正：

- `source_actor` 完全退出正式论文实验，不占用任何新正式 seed；
- 主要 baseline 改为项目冻结的 **strong nominal MPPI**；
- 用一个预注册的 `2×2` 区组设计分离“学习引导”和“动态时序风险处理”的主效应及交互；
- 再用两个局部消融解释 same-cycle filter 与 Pareto forward commit；
- 上一版 350 个 episode 只保留为开发/失败记录，不进入新实验效应估计，也不用于选择正式 seed。

在本草案、配置、runner、分析器、seed registry 和哈希均审核冻结前，不启动任何正式运行。

## 1. 研究问题与可证伪主张

### 1.1 主要研究问题

在完全相同的单动态障碍 MuJoCo 场景、因果观测、物理参数和 rollout 预算下，完整方法是否相对 strong nominal MPPI：

1. 提高无碰撞任务成功率；
2. 不增加碰撞风险；
3. 不以显著更严重的停滞、前后振荡或实时性退化换取成功率。

### 1.2 机制问题

1. 学习引导包 `G` 是否提供独立收益；
2. 动态时序风险包 `T` 是否提供独立收益；
3. `G×T` 是否存在正交互，还是两个模块收益重叠；
4. same-cycle guided-cost filter 与 Pareto forward commit 是否分别提供可复现贡献。

### 1.3 明确允许出现的负结论

以下结果均必须原样报告：

- Full 的成功率不优于 baseline；
- Full 出现 baseline 没有的碰撞或安全间距退化；
- ID 有效但 OOD 退化；
- 学习引导、时序风险包或二者交互为零或负；
- 消融差异不显著；
- 600 rollouts 下无法满足 100 ms 控制周期。

## 2. 独立实验单位与共同环境

独立实验单位是一个完整的 `simulation seed`。控制周期不是独立样本，同一 seed 下不同 arm 也不是独立样本。

所有 arm 共享：

- MuJoCo 差速小车、执行器、接触参数和传感器参数；
- 单一动态障碍的同一条 seeded 轨迹与随机流；
- `control_dt = 0.10 s`、`max_steps = 400`；
- `horizon = 36`；
- 每个有效决策严格 `600 rollouts`；
- 相同目标、Polyline、初始状态、速度/角速度约束和 scan guard；
- 相同 LaserScan、里程计和传感器噪声；
- 相同 MuJoCo 版本、Python 环境、CPU 线程设置和运行机器；
- 禁止使用 MuJoCo 障碍 identity、未来轨迹、未来 mode/event 或 future truth 作为控制输入。

共同环境从动态单障碍配置冻结；baseline 只替换控制器相关开关，不能退回 `mujoco_strong_mppi_baseline.yaml` 中的静态场景。

## 3. 正式 arms

### 3.1 两个核心因素

`G`：学习引导包。

- 冻结 Actor checkpoint；
- value-aligned ICODE / reliability-adaptive HSS；
- learned sampling prior；
- same-cycle guided-cost filter。

`T`：动态时序风险包。

- causal online obstacle tracker；
- online probability forecast；
- vetted temporal emergency lattice；
- traversal admission/commit/abort/retreat/rearm；
- Pareto low-risk forward permission。

### 3.2 四个核心 arm：完整 `2×2`

| Arm | G | T | 角色 | 必要约束 |
|---|---:|---:|---|---|
| `B00_strong_nominal_mppi` | 0 | 0 | 正式 baseline | nominal prediction；RL off；memory off；goal-warm-start/标准 MPPI；仅保留共同 reactive scan guard |
| `B10_guidance_only` | 1 | 0 | 学习引导单因素 | 完整 G；无 tracker forecast 与 temporal package；保留 reactive scan guard |
| `B01_temporal_only` | 0 | 1 | 时序风险单因素 | 标准 MPPI sampling；完整 T；无 Actor/ICODE/HSS learned guidance |
| `B11_full_proposed` | 1 | 1 | 完整方法 | 完整 G 与 T |

`B00` 的控制器合同以 `configs/research/mujoco_strong_mppi_baseline.yaml` 和 `experiments/run_mujoco_baseline_benchmark.py` 为权威来源：`prediction_mode=nominal`、`rl.enabled=false`、`memory.enable=false`。其动态场景、传感器、步数和预算由共同协议覆盖。

### 3.3 两个局部消融 arm

| Arm | 相对 Full 的唯一变化 | 目的 |
|---|---|---|
| `A_same_cycle_off` | 关闭 same-cycle guided-cost filter | 判断 learned proposals 的同周期筛选是否有效 |
| `A_pareto_commit_off` | 关闭 Pareto low-risk forward commit | 判断穿行果断性是否来自该权限 |

局部消融不得同时改变 Actor、tracker、forecast、候选数量、阈值、horizon 或速度约束。

## 4. 样本量与功效

### 4.1 主比较

主比较为 `B11_full_proposed` vs `B00_strong_nominal_mppi` 的配对成功率。

- 最小重要效应 `SESOI`：成功率绝对提高 `10` 个百分点；
- 双侧 `alpha = 0.05`；
- 目标功效 `90%`；
- 精确 McNemar 功效按配对总不一致率 `q` 做敏感性分析。

采用 `N = 360` 个全新正式 seed，ID/OOD 各 `180`。在成功率差 `+0.10` 时，精确功效为：

| 配对不一致率 q | N=360 功效 |
|---:|---:|
| 0.15 | 0.999 |
| 0.20 | 0.990 |
| 0.25 | 0.965 |
| 0.30 | 0.928 |
| 0.35 | 0.884 |

因此本设计在 `q≤0.30` 时达到至少 90% 功效；`q=0.35` 是预注册敏感性边界，不得在结果出来后改写。

### 4.2 局部消融

两个局部消融各使用预先抽取的 `120` 个正式 seed，ID/OOD 各 `60`，Full 在这些 seed 上的结果直接复用。

- outcome-aware efficiency 的 SESOI：`10 steps = 1.0 s`；
- 保守配对 SD：`30 steps`；
- 目标功效 `90%`；
- 两个消融对比按 Holm 控制家族错误率，最坏情况近似 `alpha=0.025`；
- 正态近似需求为约 `112` 对，向上取整为 `120` 对。

### 4.3 总运行量

- 四个核心 arm：`360 × 4 = 1440 jobs`；
- 两个局部消融：`120 × 2 = 240 jobs`；
- 总计：`1680 complete episode jobs`。

按上一版 artifact 密度估计，原始结果约 `1.4 GiB`。运行时间只作资源规划，不得作为中途停止依据。

## 5. Seed、区组与运行顺序

- 使用 `360` 个从未用于开发、可视化、调参或上一版确认实验的全新 sealed seeds；
- ID 与 OOD 各 180，seed registry 在运行前冻结并写入 SHA256；
- 每个 seed 构成一个完整区组，四个核心 arm 都在同一 seed 上运行；
- 两个局部消融的 120-seed 子集在开封 outcome 前由独立 schedule seed 抽取；
- seed block 顺序随机化；四核心 arm 在每个 split 内使用平衡的 Williams/Latin 顺序；
- 消融位置同样平衡，不能总在 Full 之后运行；
- 同一 seed 的所有 arm 使用 common random numbers；
- 正式实验只允许一个计算 worker，禁止同时运行 MuJoCo viewer、三障碍实验、训练或其他 CPU/GPU 重负载任务；
- BLAS、Torch 与运行器线程数冻结并记录，禁止为某个 arm 单独调整线程数。

## 6. 主要终点、约束与行为指标

### 6.1 共同主要终点

1. **任务成功**：400 步内到达 `goal distance ≤ 0.30 m`，且无碰撞、无 boundary violation；
2. **碰撞安全非劣**：Full 相对 baseline 的配对碰撞风险差，其单侧 95% 上界小于 `+0.02`。

只有以下条件同时成立，才能写“Full 相对 baseline 获得确认性改善”：

- 成功率差 `Full - Baseline > 0`，双侧 exact McNemar `p<0.05`，且 95% CI 下界大于 0；
- pooled 碰撞非劣性通过；
- ID 与 OOD 各自的成功率差单侧 95% CI 下界均高于 `-0.05`；
- ID 与 OOD 各自的碰撞风险差单侧 95% CI 上界均低于 `+0.05`；
- 完整性与实时性 gate 通过。

### 6.2 关键次要终点

- failure-penalized control steps：成功使用实际 steps；失败固定记为 400；
- time-to-goal（仅在双方均成功的配对中报告，并同时报告选择比例）；
- final goal distance；
- maximum path progress；
- minimum clearance；
- boundary violation；
- control jerk、轨迹长度、stuck steps、spin steps。

### 6.3 针对已观察问题的预注册行为指标

以下指标在分析代码中运行前固定定义：

- **方向切换次数**：`applied_v` 穿越 `+0.05/-0.05 m/s`，且新方向至少持续 2 个周期；
- **三相振荡**：2.0 s 窗口内出现 `F-R-F` 或 `R-F-R`；
- **过早倒车**：倒车段开始时 clearance `≥0.80 m`，且 TTC `>1.50 s`；
- **closing 清除后的释放延迟**：closing 连续 2 周期为 false 后，到首次连续 2 周期 `v>0.05 m/s` 的时间；
- **冲突区穿行耗时**：进入预注册 conflict-zone entry 后到超过 exit 的时间；未退出者记为删失/失败，不得只统计成功样本；
- **零速暴露**：hard risk 或 raw closing 活跃期间 `|v|≤0.05 m/s` 的周期数。

阈值来自控制周期和现有安全合同，不得在查看正式轨迹后修改。

### 6.4 实时性与预算 gate

- 每个有效决策、每个 arm 必须恰好记录 600 rollouts；
- Full 的 pooled、ID、OOD decision-level P95 均 `<100 ms`；
- 同时报告 P50/P95/P99、deadline miss fraction 和每 episode P95；
- deadline miss fraction 必须 `<5%`；
- baseline 与全部消融也完整报告实时性，但不能通过降低其 rollout 数量制造 Full 的优势；
- 组件 profiler 在资格测试中启用；正式计时是否启用必须对所有 arm 一致，并在冻结协议中明确。

## 7. 统计分析

### 7.1 主比较

- 成功与碰撞：按 seed 配对，报告 2×2 discordance table、风险差与置信区间；成功使用双侧 exact McNemar；
- 碰撞非劣性：配对风险差的单侧 95% CI，与 `+0.02` margin 比较；
- 成功优效和碰撞非劣是 intersection-union 条件，二者都必须通过，不用以其中一个补偿另一个；
- 连续终点报告 seed-level paired mean/median、配对 t CI、Wilcoxon 敏感性分析和 200,000 次 whole-seed bootstrap CI；
- 不把控制 timestep 当独立样本。

### 7.2 `2×2` 核心机制分析

对每个 seed 计算：

- `G main effect = [(B10-B00) + (B11-B01)] / 2`；
- `T main effect = [(B01-B00) + (B11-B10)] / 2`；
- `G×T interaction = B11-B10-B01+B00`。

连续/风险差尺度上的效应均以 whole-seed bootstrap 给出 95% CI；三个机制检验使用 Holm 校正。ID/OOD interaction 预注册为分层泛化分析，不用 pooled 正结果掩盖 OOD 伤害。

### 7.3 局部消融

- `B11 - A_same_cycle_off`；
- `B11 - A_pareto_commit_off`。

两个对比使用 Holm 校正。若差异不显著，只能写“本实验未确认独立贡献”，不能按均值方向包装。

## 8. 资格测试、冻结与完整性

### 8.1 正式运行前资格测试

使用不属于正式 registry 的 16 个开发 seed（ID/OOD 各 8），仅检查：

- 六个 arm 均可启动并自然退出；
- schema、指标和行为计数器完整；
- baseline 确实 `RL off / memory off / nominal prediction`；
- Full 与消融开关仅产生预期差异；
- 无 MuJoCo future truth 泄漏；
- 每周期 600 rollouts；
- 单 worker 下实时性具备进入正式实验的工程可行性；
- artifact 路径、恢复逻辑和哈希审计正常。

资格测试不得用于论文效应估计。若资格测试失败，可在开发区修复；修复后重新资格测试并重新冻结，不能带着已打开的正式 outcome 调参。

### 8.2 冻结清单

正式启动前必须冻结并记录 SHA256：

- common environment config；
- 六个 arm 的 resolved-config delta；
- runner、analysis、metrics、MPPI、safety arbiter、tracker/forecast 代码；
- Actor 与 ICODE/value checkpoint；
- seed registry 与 schedule；
- Python/MuJoCo/Torch/Numpy 版本、Windows 版本和硬件清单；
- 本预注册文件。

### 8.3 中途查看与停止规则

- 完成 1680/1680 前只查看 progress、进程存活、stderr 和文件完整性，不查看 outcome；
- 不按中间趋势提前停止，不替换失败 seed，不补“有利 seed”；
- 基础设施崩溃只允许重跑相同 seed/arm/config，并保留第一次失败 provenance；
- 算法碰撞、停滞、boundary violation、超时和实时性失败均是有效结果，不重跑；
- 任一必要 episode 缺失则正式矩阵标记 incomplete，不做插补。

## 9. 论文结果表的预定结构

主表包含四个核心 arm，按 pooled / ID / OOD 报告：

- success rate；
- collision rate；
- failure-penalized steps；
- final goal distance；
- minimum clearance；
- direction switches / three-phase oscillations；
- planner P95 与 deadline misses。

机制表报告 `G`、`T`、`G×T` 三个预注册对比；局部消融表报告两个 Holm-adjusted 对比。每个表同时给出效应量、95% CI、原始 p 值和校正 p 值，不只给均值。

## 10. 需要用户审核的冻结决策

在实施前需逐项确认：

1. 是否同意 `Source Actor` 完全退出正式矩阵；
2. 是否同意以 strong nominal MPPI 作为唯一主要 baseline；
3. 是否同意四核心 arm 的 `G×T` 设计；
4. 是否同意两个局部消融：same-cycle filter 与 Pareto commit；
5. 是否同意主样本量 360（ID/OOD 各 180）；
6. 是否同意局部消融各 120 对；
7. 是否同意成功率 `+10 pp` 的 SESOI；
8. 是否同意碰撞非劣 margin `+2 pp`；
9. 是否同意 ID/OOD 分层成功伤害 margin `-5 pp` 和碰撞伤害 margin `+5 pp`；
10. 是否同意单 worker、无并发重负载的正式执行合同；
11. 是否同意先完成并审核单障碍结果，再单独预注册三障碍实验。

任何一项未批准，本草案保持 DRAFT，不生成正式 seed schedule，不启动实验。
