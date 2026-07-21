# 截至 L245 的当前实验数据详细汇总

**整理日期：** 2026-07-21  
**项目：** mobile-robot-mppi-study  
**当前分支：** `codex/post-l233-five-direction-experiments`  
**覆盖范围：** L243 教师数据与 BC+SAC 训练、冻结选模、L242 历史对照、L244/L245 MuJoCo 闭环开发实验  
**证据状态：** `ANALYZED`（已完成数据与来源审计，但尚未完成多 seed sealed 重复验证）  
**总体置信等级：** `CAUTION`（结果真实且可追溯，但闭环 Tracking 目前只有一个 development seed）

---

## 1. 先给出结论

截至 L245，当前数据给出的最准确结论是：

1. 已经建立了一条完整、可追溯的 **安全教师数据 → BC 锚定的 SAC Actor → 冻结验证选模 → MuJoCo 闭环 MPPI** 实验链路。
2. L243 训练后，冻结验证集选中的模型由 `global_step=0` 前进到 `global_step=30000`，验证成功数由此前的 **2/30 提高到 6/30**；说明训练不是空跑，且学习后的 Actor 在冻结验证规则下优于初始化模型。
3. L244 中，BC 锚定 Actor 接入完整方法后，三场景平均路径完成度由 L242 的 **0.1287 提高到 0.2118**，绝对增加 **0.0832**；Hairpin 场景由 **0.1077 提高到 0.3422**，是当前最明显的正向闭环结果。
4. L244 三个场景均 **零碰撞**，RL proposal authority 为非零，说明 Actor 的建议确实进入了 MPPI，而不是被门控完全关闭。
5. 但 L244 在 S-Chicane 和 Infinity 各出现 1 个 boundary violation step，且三个场景在 700 步内均未完成整条路径，因此严格 Development Gate 未通过。
6. L245 仅把 path boundary buffer 从 0.05 m 改为 0.075 m。它没有消除 boundary violation，反而使 Hairpin 完成度从 0.3422 降到 0.2589。因此 L245 证明：**简单继续增大 buffer 不是有效修复方向**。
7. 这些数据支持“路径/残差条件 Actor 能够改善局部推进，并能与 ICODE-MPPI 发生实际耦合”这一较窄结论；尚不能支持“完整方法已在复杂路径上稳定优于全部基线”或“已经具备论文正式主结果”的强结论。

本汇总遵照当前要求，**不新增消融实验**。因此也不把 L244 的改善单独归因于 BC 或 SAC：当前模型是 **BC 锚定的 SAC Actor**，闭环提升是这套混合训练及完整控制链共同产生的结果。

---

## 2. Material Passport（实验材料身份证）

| 项目 | 内容 |
|---|---|
| 仿真平台 | MuJoCo 3.2.3 |
| 机器人/控制 | 差速移动机器人，控制量为归一化的 `(v, omega)` |
| 核心方法 | Value-Consistent ICODE + Path/Residual-Conditioned RL Prior + role-aware Reliability-Weighted Value/HSS + MPPI |
| 学习 Actor | SAC Actor，训练中加入行为克隆（BC）锚定项 |
| 学生观测 | 69 维；包含 LaserScan 压缩、运动状态、残差上下文、安全状态与路径预览；不包含绝对位姿 |
| Actor 输出 | 2 维 direct control proposal，随后作为 MPPI 引导信息使用，并非直接绕过 MPPI 与安全链执行 |
| 教师特权信息 | 教师采集时可访问 ground-truth pose 与离线安全参考路径；这些信息只保存在审计文件，不进入学生在线观测 |
| Tracking 场景 | Hairpin、S-Chicane、Infinity |
| 闭环物理域 | `nominal_seen` |
| 闭环 development seed | `923301001` |
| MPPI 等预算 | 总候选预算 K=100；Full 为 50×2 iterations，ICODE 对照为 100×1 iteration |
| 最大闭环步数 | 700 |
| 当前结果级别 | development qualification，不是 sealed 正式实验 |

---

## 3. L243 教师数据集

### 3.1 数据是怎样产生的

教师控制器 `ScriptedPolylineDirectControl` 使用预先设计的安全参考路径和 MuJoCo ground-truth pose 产生归一化 `(v, omega)` 控制。每一步同时记录学生实际能够获得的 69 维观测和教师动作。

关键边界如下：

- 教师可以使用 ground-truth pose 和参考路径，这是训练期特权信息。
- 学生观测来自正常 `DirectControlEnv.reset_and_step` 链路。
- 学生观测明确设置 `include_absolute_pose=false`，因此在线 Actor 不会直接读取真值绝对位姿。
- 教师的特权路线、waypoints 和 simulator truth 只写入 `audit/`，不混入训练 shard。
- 训练动作语义为 `normalized_direct_control_v_omega`，输入/输出契约为 69D → 2D，避免此前 local-subgoal 教师与 direct policy 的语义混用。

### 3.2 总体数量

| 指标 | 数值 |
|---|---:|
| 请求回合数 | 90 |
| 安全成功回合 | 60 |
| 失败回合 | 30 |
| 进入训练 shards 的失败回合 | 0 |
| 可训练 transition 样本 | 41,932 |
| 场景数 | 6 |
| 每场景 seeds | 15 |

失败回合没有被删除：它们保留在 `audit/episodes.csv` 及逐回合审计工件中，只是依照冻结的数据契约，不作为安全教师示范写入训练 shards。

### 3.3 分场景结果

| 场景 | 请求回合 | 安全成功 | 失败 | 进入 shards 的样本数 |
|---|---:|---:|---:|---:|
| `l222_serpentine_safe` | 15 | 15 | 0 | 21,248 |
| `l218_giant_u` | 15 | 15 | 0 | 5,693 |
| `l218_opposed_u` | 15 | 0 | 15 | 0 |
| `l222_nested_u_safe` | 15 | 15 | 0 | 7,699 |
| `l218_cylinder_forest` | 15 | 0 | 15 | 0 |
| `l222_cylinder_spiral_safe` | 15 | 15 | 0 | 7,292 |
| **合计** | **90** | **60** | **30** | **41,932** |

这说明数据集对四张地图提供了有效安全示范，但在 `opposed_u` 与 `cylinder_forest` 上教师系统性失败。因而当前 Actor 的示范覆盖并不是六地图均衡覆盖，不能把它表述为“已学会全部复杂地图”。

### 3.4 按 seed 拆分

数据没有按单 timestep 随机打散，而是按 seed 拆分，避免同一轨迹泄漏到多个集合。

| Split | 回合数 | 样本数 | Seeds | 文件 SHA256 |
|---|---:|---:|---|---|
| Train | 36 | 25,159 | 20262401, 02, 03, 07, 09, 10, 13, 14, 15 | `48c827dd...214a3` |
| Validation | 12 | 8,390 | 20262404, 11, 12 | `f28b1f47...1a937` |
| Test | 12 | 8,383 | 20262405, 06, 08 | `635087fd...43746` |

完整哈希、字节数和配置见：

```text
results/research_platform/rl/l243_direct_control_teacher_dataset_v1/manifest.json
```

---

## 4. L243 BC 锚定 SAC Actor 训练

### 4.1 这是不是监督学习

其中 **BC loss 是监督学习项**，但整体训练不是纯监督学习。当前 Actor 的更新目标包含两部分：

```text
SAC 强化学习目标 + BC 教师动作锚定目标
```

因此更准确的称呼是 **BC-anchored SAC**，即“带行为克隆锚定的强化学习”。BC 用来给 Actor 一个安全且有方向的初始行为偏置；SAC 仍通过环境回报、critic 和 replay buffer 更新策略。由于本阶段没有新增 BC-only 消融，本报告不试图区分两种学习信号各自贡献了多少。

### 4.2 三个独立训练运行

| 训练 seed | 环境步数 | 更新记录 | 验证记录 | Checkpoint 数 | NaN/Inf |
|---:|---:|---:|---:|---:|---:|
| 20262331 | 30,000 | 29,001 | 210 | 9 | 0 |
| 20262332 | 30,000 | 29,001 | 210 | 9 | 0 |
| 20262333 | 30,000 | 29,001 | 210 | 9 | 0 |

训练使用的 validation bases 分别为 20262931、20262932、20262933。三组训练均为 CPU 运行；当前 WSL PyTorch 为 CPU-only，因此不能把本阶段写成 CUDA 训练。

训练 provenance：

- Git SHA：`263b58bbe14a797656a8433be453726cf255eafa`
- 源 Actor SHA256：`bf26a67ebac313930d63760db931e5d50704cdd9181923afd7f66d16159356b4`
- 教师数据 manifest fingerprint：`918712d513763fd272db25d7a428c6fcf563e6ca1b98c0f089cb46f52cc91fbb`
- Actor 输入/输出：69D direct observation → 2D direct action
- normalizer：冻结；Actor、critics 与 SAC 状态参与更新

---

## 5. 冻结验证选模

候选模型只能通过预先冻结的 validation 字典序选择，不能读取 L234/L244 的 development 闭环结果后再挑 checkpoint。

选择顺序为：

1. 最少碰撞；
2. 最多成功；
3. 最大平均路径完成度；
4. 最小 cross-track RMSE；
5. 最小终点距离；
6. 最大平均 return；
7. seed 与 step 仅用于最终稳定 tie-break。

### 5.1 选中模型

| 项目 | 数值 |
|---|---|
| 候选数 | 21 |
| 选中训练 seed | 20262332 |
| 选中 global step | 30,000 |
| 验证回合 | 30 |
| 验证碰撞 | 0 |
| 验证成功 | 6/30 |
| 验证平均完成度 | 0.3843 |
| 验证 cross-track RMSE | 0.2901 m |
| 验证平均终点距离 | 3.6355 m |
| 验证平均 return | -1075.1344 |
| Checkpoint SHA256 | `e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c` |

此前 L241 冻结规则选中的初始化模型为 `global_step=0`，验证结果为 0 碰撞、2/30 成功。L243 选中 30k 模型并达到 6/30，说明训练产生了可测的验证集改善，而不是仅改变文件或日志。

---

## 6. 闭环指标解释

| 指标 | 含义 | 趋势 |
|---|---|---|
| `path_completion_ratio` | 已沿参考路径完成的比例，0 到 1 | 越高越好 |
| `success` | 是否在步数上限内完成任务 | 1 为成功 |
| `collision` | 是否发生物理碰撞 | 0 为好 |
| `cross_track_rmse` | 机器人相对参考中心线的横向误差均方根 | 越低越好 |
| `boundary_violation_steps` | footprint 超出允许路径走廊的步数 | 0 为好 |
| `minimum_footprint_boundary_margin` | footprint 距走廊边界的最小余量 | 非负为好 |
| `safety_interventions` | 安全层介入控制的次数 | 一般越少越好，但需与进度共同解释 |
| `proposal_authority_mean` | RL proposal 在在线 guidance 中获得的平均权限 | 非零说明 RL 真正参与 |
| `guided_elites` | MPPI 最终精英样本中来自引导提议的数量 | 非零说明 proposal 进入有效采样 |
| `planner_compute_ms_mean` | 单次规划平均耗时 | 越低越好 |

需要特别注意：完成度、cross-track 与安全不是同一指标。某方法可以推进更快，但走得更偏；也可能非常贴合中心线，却因安全仲裁频繁而推进缓慢。

---

## 7. L242 历史闭环基准

L242 使用冻结的 selected-initial Actor，作为 L244 的同 seed、同地图、同预算历史对照。

| 场景 | 完成度 | 成功 | 碰撞 | Cross-track RMSE | Boundary steps | 最小边界余量 | 安全介入 | RL authority | Guided elites | Planner ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hairpin | 0.1077 | 0 | 0 | 0.4222 | 0 | 0.0215 | 111 | 0.0922 | 259 | 508.5 |
| S-Chicane | 0.1758 | 0 | 0 | 0.2527 | 1 | -0.0057 | 27 | 0.0837 | 239 | 437.4 |
| Infinity | 0.1025 | 0 | 0 | 0.3083 | 0 | 0.0998 | 166 | 0.0610 | 110 | 494.8 |

汇总：平均完成度 **0.1287**，成功 0/3，碰撞 0/3，boundary steps 合计 1，安全介入合计 304。

---

## 8. L244 BC 锚定 Actor 闭环结果

### 8.1 分场景结果

| 场景 | 方法 | 完成度 | 成功 | 碰撞 | Cross-track RMSE | Boundary steps | 最小边界余量 | 安全介入 | RL authority | Guided elites | Planner ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hairpin | L244 BC Full | **0.3422** | 0 | 0 | 0.4333 | 0 | 0.0072 | 26 | 0.1759 | 30 | 390.5 |
| Hairpin | L244 ICODE | 0.1118 | 0 | 0 | **0.0784** | 0 | 0.5630 | 442 | — | 0 | 131.3 |
| S-Chicane | L244 BC Full | **0.1877** | 0 | 0 | 0.4117 | 1 | -0.0027 | 90 | 0.0755 | 25 | 400.0 |
| S-Chicane | L244 ICODE | 0.1827 | 0 | 0 | **0.1288** | 0 | 0.5618 | 437 | — | 0 | 129.9 |
| Infinity | L244 BC Full | 0.1055 | 0 | 0 | 0.2823 | 1 | -0.0072 | 85 | 0.0961 | 20 | 391.0 |
| Infinity | L244 ICODE | **0.1084** | 0 | 0 | **0.0883** | 0 | 0.5683 | 416 | — | 0 | 128.1 |

表中 ICODE 的 `proposal_authority` 不用于解释 RL，因为 ICODE-MPPI 本身没有 Actor proposal；原始记录中的占位值不应被误读为 RL authority。

### 8.2 相对 L242 的变化

| 场景 | L242 Full | L244 BC Full | 完成度绝对变化 |
|---|---:|---:|---:|
| Hairpin | 0.1077 | 0.3422 | **+0.2345** |
| S-Chicane | 0.1758 | 0.1877 | +0.0119 |
| Infinity | 0.1025 | 0.1055 | +0.0030 |
| **三场景平均** | **0.1287** | **0.2118** | **+0.0832** |

其他汇总变化：

- 碰撞：0 → 0；
- boundary steps：1 → 2；
- 安全介入：304 → 201；
- BC Full 平均 proposal authority：0.1158；
- guided elites 合计：75；
- BC Full 平均 planner time：393.8 ms；
- 同轮 ICODE 平均 planner time：129.8 ms。

### 8.3 L244 的正确解释

L244 给出了明确但局部的正向信号：

- Actor 的 proposal 被实际采用；
- Hairpin 的路径推进显著提高；
- 三场景平均完成度提高；
- 没有发生物理碰撞；
- 安全介入次数减少。

但 L244 也暴露了三个问题：

- 三场景均未在 700 步内完成整条路径；
- boundary steps 从 1 增至 2；
- cross-track 并未相对 ICODE 全面改善。

因此 L244 是“**明显的 development 正向结果，但严格 Gate 未通过**”，不能写成全面胜出。

---

## 9. L245：7.5 cm Boundary Buffer 单变量探针

L245 只把 `path_boundary_buffer` 从 0.05 m 增加到 0.075 m，其他核心方法、Actor、地图、seed、预算和安全链不变。

| 场景 | 完成度 | 相对 L244 | 相对 L242 | 成功 | 碰撞 | Cross-track RMSE | Boundary steps | 最小边界余量 | 安全介入 | RL authority | Guided elites | Planner ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hairpin | 0.2589 | **-0.0833** | +0.1512 | 0 | 0 | 0.4278 | 0 | 0.0031 | 73 | 0.1554 | 30 | 393.3 |
| S-Chicane | 0.1816 | -0.0061 | +0.0058 | 0 | 0 | 0.4607 | 1 | -0.0097 | 37 | 0.0953 | 25 | 409.0 |
| Infinity | 0.1061 | +0.0005 | +0.0036 | 0 | 0 | 0.3340 | 1 | -0.0067 | 92 | 0.1054 | 20 | 400.1 |

汇总：

- 平均完成度：**0.1822**；
- 相对 L242 平均增加：**+0.0535**；
- 相对 L244 平均下降：**-0.0296**；
- 碰撞：0/3；
- boundary steps：2，未改善；
- 平均 RL authority：0.1187；
- guided elites：75。

L245 Gate 失败有两个直接原因：

1. S-Chicane 与 Infinity 的 boundary violation 没有消失；
2. Hairpin 相对 L244 回退 0.0833，超过预注册允许的 0.02。

因此 L245 的价值主要是排除了一个看似直观、实际无效的工程修复：**不能依靠继续增大 buffer 来同时获得安全与推进改善**。

---

## 10. 跨阶段汇总

| 阶段/方法 | 平均完成度 | 成功 | 碰撞 | Boundary steps | 安全介入 | RL authority | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| L242 selected-initial Full | 0.1287 | 0/3 | 0/3 | 1 | 304 | 0.0790 左右 | 历史基准 |
| L244 BC-anchored Full | **0.2118** | 0/3 | 0/3 | 2 | **201** | **0.1158** | 明显提高局部推进，但安全 Gate 未通过 |
| L244 ICODE-MPPI | 0.1343 | 0/3 | 0/3 | 0 | 1295 | 不适用 | 跟踪误差小但推进慢，规划计算更低 |
| L245 BC Full, buffer 0.075 | 0.1822 | 0/3 | 0/3 | 2 | 202 | 0.1187 | 比 L242 好，但不如 L244；buffer 修复无效 |

注：L244 ICODE 安全介入合计为 442+437+416=1295。该数字与方法推进速度和安全仲裁交互有关，不能单独解读为碰撞风险。

---

## 11. 当前证据支持什么

当前数据可以支持以下陈述：

1. 69D path/residual-conditioned Actor 的数据、训练、加载、选模和 MuJoCo 闭环接入已经打通。
2. 学习后的 checkpoint 在冻结 validation 规则下优于初始化 checkpoint：成功数 2/30 → 6/30。
3. L244 的 Actor guidance 不是形式上的开关：proposal authority 和 guided elites 均为非零。
4. 在 Hairpin development 场景中，BC-anchored Actor 明显提高了路径推进；三场景平均完成度也提高。
5. 改善是在零物理碰撞条件下取得的。
6. 单纯把 boundary buffer 从 0.05 m 增加到 0.075 m 不能解决当前边界问题。

---

## 12. 当前证据不能支持什么

当前数据不能支持以下强结论：

1. 不能说完整方法已经在三个 Tracking 场景上成功完成任务，因为成功率仍为 0/3。
2. 不能说完整方法稳定优于 ICODE-MPPI，因为仅 Hairpin 有大幅完成度优势，Infinity 略低于 ICODE，且 cross-track 普遍高于 ICODE。
3. 不能声称具有跨 seed 泛化性；L242/L244/L245 闭环比较只有一个 development seed。
4. 不能把 L244 的全部增益归因于“强化学习”或“BC”其中之一；本轮没有新增 BC-only / SAC-only 消融。
5. 不能将这些 development qualification 结果写成 sealed 正式论文结果。
6. 不能声称教师数据覆盖六张地图的成功行为；其中两张地图没有合格教师片段。
7. 不能声称 CUDA 加速；本阶段 WSL PyTorch 为 CPU-only。

---

## 13. 统计与证据风险审计

当前没有进行显著性检验、置信区间或 seed-cluster bootstrap，因为闭环仅有一个 seed；在这种样本结构下报告 p 值会产生虚假的确定性。

### 13.1 十一类常见推断谬误扫描

| 风险 | 当前判断 |
|---|---|
| Simpson's paradox | 三场景平均增益主要由 Hairpin 驱动；必须同时报告分场景与平均值，不能只展示平均值。 |
| Ecological fallacy | 不能由三场景平均提升推出每个场景都提升，也不能推出每个 seed 都会提升。 |
| Berkson's paradox | 训练 shards 只包含安全成功教师片段，训练数据是条件筛选后的分布；需保留并报告 30 个失败回合。 |
| Collider bias | 教师“成功且安全”同时受地图难度、教师控制和安全链影响；在选中样本上分析可能制造虚假关联。 |
| Base-rate neglect | 教师总体成功率为 60/90，而不是只看 41,932 个成功 transition；两张地图成功率为 0/15。 |
| Regression to the mean | 单个 seed 的大幅 Hairpin 改善可能含随机波动，需要多 seed 复现。 |
| Survivorship bias | 成功片段进入训练 shard，但失败审计必须保留；本报告已同时列出失败数与失败地图。 |
| Look-elsewhere effect | L236-L245 存在多次透明 development 尝试；不能把最终看起来最好的 L244 当作未调参的独立确认。 |
| Garden of forking paths | 多轮单变量开发虽然有预注册协议，但仍属于开发阶段；正式结论必须使用后续冻结 sealed 配置。 |
| Correlation ≠ causation | L244 改善与 BC-anchored Actor 同时出现，但没有消融，不能证明是某一损失项单独造成。 |
| Reverse causality | safety interventions 减少可能源于更快推进、轨迹变化或门控行为，不能直接推断“安全能力提高导致介入减少”。 |

---

## 14. 可复现性与原始证据索引

### 14.1 教师数据与训练

```text
results/research_platform/rl/l243_direct_control_teacher_dataset_v1/
results/research_platform/rl/l243_path_preview_bc_anchor_seed20262331_30k_v1/
results/research_platform/rl/l243_path_preview_bc_anchor_seed20262332_30k_v1/
results/research_platform/rl/l243_path_preview_bc_anchor_seed20262333_30k_v1/
results/research_platform/rl/l243_path_preview_bc_anchor_selection_v1/
```

### 14.2 闭环结果

```text
results/research_platform/rl/tracking_l244_bc_anchor_hairpin_seed923301001/
results/research_platform/rl/tracking_l244_bc_anchor_s_chicane_seed923301001/
results/research_platform/rl/tracking_l244_bc_anchor_infinity_seed923301001/
results/research_platform/rl/tracking_l245_buffer075_hairpin_seed923301001/
results/research_platform/rl/tracking_l245_buffer075_s_chicane_seed923301001/
results/research_platform/rl/tracking_l245_buffer075_infinity_seed923301001/
```

### 14.3 配置、协议、表和图

```text
configs/research/tracking_path_actor_bc_gate_l244.yaml
configs/research/tracking_path_actor_bc_buffer075_probe_l245.yaml
docs/experiments/post_l217/l243_path_actor_bc_anchor_training_report.md
docs/experiments/post_l217/l244_path_actor_bc_tracking_gate_report.md
docs/experiments/post_l217/l245_bc_actor_boundary_buffer_report.md
docs/experiments/post_l217/tables/l244_tracking_gate_summary.csv
docs/experiments/post_l217/tables/l245_boundary_buffer_probe_summary.csv
docs/experiments/post_l217/figures/fig_l244_tracking_gate.png
docs/experiments/post_l217/figures/fig_l245_boundary_buffer_probe.png
```

### 14.4 关键提交

| Commit | 作用 |
|---|---|
| `263b58b` | 记录 L243 BC smoke gate |
| `edf9e1a` | 冻结 L244 BC Actor Tracking Gate |
| `24e5d02` | 报告 L244 并预注册 L245 |
| `83647a5` | 冻结 L245 boundary buffer 结果 |

最近一次相关测试记录为：**36 passed，约 2.67 s**。该测试说明代码级接口和相关回归通过，不等同于统计学验证。

---

## 15. 面向论文的当前状态判断

当前证据已经超过“代码能跑”的阶段，进入“具有真实正向信号的开发实验”阶段。最有价值的数据是：

```text
训练选模：validation success 2/30 → 6/30
闭环推进：mean completion 0.1287 → 0.2118
Hairpin：0.1077 → 0.3422
碰撞：0 → 0
RL proposal authority：非零
```

但论文主结果仍缺少：多 seed、冻结 sealed protocol、完整成功率、边界约束稳定性和更强统计证据。因此当前材料适合用于：

- 方法开发汇报；
- 解释为什么引入 BC 锚定的 SAC Actor；
- 形成后续正式实验的工程与数据依据；
- 作为负向结果与方法演进记录。

当前不适合直接写成“完整方法已经全面胜出”的论文结论。

---

## 16. 一句话总结

**截至 L245，我们已经证明 BC 锚定的路径/残差条件 SAC Actor 能够真实进入 ICODE-MPPI 的在线采样并显著改善 Hairpin 与平均路径推进，但复杂 Tracking 尚未实现稳定全程成功，边界约束仍是进入 sealed 正式实验前的主要瓶颈。**

