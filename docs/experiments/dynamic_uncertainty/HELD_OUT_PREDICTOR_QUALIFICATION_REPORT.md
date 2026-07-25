# Held-out ID/OOD 概率预测器正式资格报告

日期：2026-07-23  
平台：Windows native  
分支：`codex/change-aware-probabilistic-mppi`  
结论：**PASS，可以结束概率预测器开发阶段，进入风险接口与 MPPI 集成阶段。**

## 1. 结论边界

本次结论只适用于冻结的 V3 动态障碍过程、冻结的 OOD 偏移和冻结的
`dual_975_999` Change-Aware IMM。实验开始后没有根据 held-out 结果修改障碍物、
预测器、阈值、seed、Gate 或排除规则。

通过资格审查表示：

- Change-Aware IMM 在未参与开发的 ID 和 OOD 轨迹上，都稳定改善概率预测质量；
- 相对普通 IMM，它没有牺牲总体位置预测精度；
- 所有轨迹物理约束、数值稳定性、因果预测和精确重放检查均通过；
- 该预测器可以作为下一阶段风险感知 MPPI 的输入。

它不表示 P2--P4 模式切换期间的概率校准已经完美，也不表示闭环避障性能已经得到
验证。闭环安全性必须在后续 MPPI 实验中单独检验。

## 2. 冻结实验设计

- 独立统计单位：完整障碍物轨迹 seed，而不是轨迹内的预测窗口。
- held-out ID：10 个新 seed × 4 个过程 × 2 个噪声档，共 80 条轨迹。
- held-out OOD：10 个新 seed × 4 个过程 × 2 个偏移档，共 80 条轨迹。
- 总计：160 条轨迹；每条轨迹上的四个预测器共享完全相同的真值与观测。
- 主要比较：P2--P4 上 Change-Aware IMM 减去 ordinary IMM 的 NLL。
- 不确定性分析：10,000 次整 seed bootstrap 置信区间。
- 显著性分析：精确双侧配对符号翻转检验，ID/OOD 两项使用 Holm 校正。

## 3. 主要结果

### 3.1 概率质量

数值越低越好。

| Split | Gaussian CV P2--P4 NLL | ordinary IMM P2--P4 NLL | Change-Aware P2--P4 NLL | Change/ordinary |
|---|---:|---:|---:|---:|
| ID | 15.6129 | 11.7631 | **8.8354** | **0.7511** |
| OOD | 22.7353 | 14.5501 | **10.9847** | **0.7550** |

Change-Aware 相对 ordinary IMM 的 P2--P4 NLL 在 ID 上降低约 24.9%，在 OOD
上降低约 24.5%。相对 Gaussian CV 的 NLL 比率分别为 0.5659 和 0.4832。

### 3.2 配对 seed 统计

差值定义为 `Change-Aware NLL - ordinary IMM NLL`，因此负值表示
Change-Aware 更好。

| Split | 10 个 seed 中改善数 | 平均差值 | 95% bootstrap CI | exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| ID | 10/10 | **-2.9277** | **[-3.2313, -2.6061]** | 0.001953 | 0.003906 |
| OOD | 10/10 | **-3.5654** | **[-3.9649, -3.1956]** | 0.001953 | 0.003906 |

直观解释：

- 10/10 表示每一个独立 seed 都指向同一个改善方向，不是由少数幸运轨迹拉动均值。
- 置信区间完全位于 0 以下，表示即使考虑不同 seed 之间的波动，平均改善仍明确为负。
- 符号翻转检验把“若两种方法其实一样，某个 seed 谁更好应当像随机正负号”作为零假设。
  10 个 seed 有 1,024 种正负组合；当前结果属于最极端的一小部分。
- Holm 校正考虑了同时检验 ID 和 OOD 两项后，两个结果仍低于预注册的 0.05 门槛。

### 3.3 精度与变化后恢复

| Split | 总体 ADE ordinary | 总体 ADE Change-Aware | ADE ratio | post 1--3 s NLL ratio |
|---|---:|---:|---:|---:|
| ID | 0.3865 m | **0.3763 m** | **0.9736** | **0.7430** |
| OOD | 0.5238 m | **0.5053 m** | **0.9647** | **0.7959** |

Change-Aware 在两个 split 上都没有以位置精度换取概率质量，反而将总体 ADE
降低约 2.6% 和 3.5%。模式改变后 1--3 s 的 NLL 也分别降低约 25.7% 和 20.4%。

### 3.4 校准结果与保留风险

| Split | ordinary P2--P4 coverage@90 | Change-Aware P2--P4 coverage@90 | Change-Aware 四过程 macro coverage@90 |
|---|---:|---:|---:|
| ID | 0.6297 | **0.6612** | **0.7422** |
| OOD | 0.5357 | **0.5797** | **0.6737** |

Change-Aware 的覆盖率比 ordinary IMM 高 5.0%（ID，相对值）和 8.2%（OOD，相对值），
满足冻结 Gate；ID 四过程 macro coverage@90 也落在预注册的 `[0.68, 0.95]`。

但是，名义 90% 区域在 P2--P4 上只覆盖 66.1%（ID）和 58.0%（OOD），说明模式切换
附近仍然偏自信。后续 MPPI 不应把该置信区域解释成已经精确校准的“真实 90% 安全
边界”。风险接口应保留显式安全裕量，并在闭环实验中报告经验碰撞率、近失碰撞率和
风险预测校准。

## 4. Gate 与完整性

- 冻结资格 Gate：16/16 通过。
- 正式运行：160/160 完成，160 个唯一实验键。
- ID/OOD：80/80 + 80/80。
- 物理与数值不变量：160/160 通过。
- 逐运行 provenance：160/160 存在。
- 逐文件 SHA-256：0 个不匹配。
- 160 份运行的概率模块源码快照：完全一致。
- stderr：空。
- 正式实验产物完整性审计：通过。

## 5. 下一阶段

下一阶段不再修改 V3 障碍物生成器或本次预测器。应当：

1. 冻结从混合高斯预测到时空碰撞风险的接口和数值审计；
2. 将该风险代价接入 MPPI；
3. 使用已经约定的起点—终点走廊，使循环、无规律运动的障碍物反复穿越机器人必经区域；
4. 先做小规模闭环 smoke，再预注册正式对照实验；
5. 对比 deterministic、Gaussian CV、ordinary IMM 和 Change-Aware IMM 驱动的
   MPPI，并同时报告安全性、任务效率、控制平滑性和计算时延。

## 6. 可复核产物

- 预注册：`docs/experiments/dynamic_uncertainty/HELD_OUT_PREDICTOR_PREREGISTRATION.md`
- 完整结果：`research_artifacts/held_out_predictor_qualification/analysis/summary.json`
- seed 级差值：`research_artifacts/held_out_predictor_qualification/analysis/seed_level_nll.csv`
- 主图：`research_artifacts/held_out_predictor_qualification/figures/held_out_predictor_qualification.png`
- 完整性审计：`research_artifacts/held_out_predictor_qualification/integrity_audit.json`

