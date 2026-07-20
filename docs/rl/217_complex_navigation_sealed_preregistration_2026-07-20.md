# L217 复杂场景封存确认实验预注册

日期：2026-07-20
状态：结果产生前冻结；提交并推送本文件后方可运行

## 1. 固定研究问题

本实验检验：在静态复杂障碍导航与未见 MuJoCo 物理失配下，以下完整方法

> Value-Consistent ICODE + Residual-Conditioned RL Prior + role-aware
> Reliability-Weighted Value/HSS + MPPI

相对于 `Simple Combination` 是否在相同 MPPI rollout 预算与不削弱安全链的条件下，保持安全到达能力并改善连续控制表现。该问题、七方法消融矩阵和核心机制均不因 L217 改变。

## 2. 开启依据及其证据边界

L216 完整 development matrix 包含七方法、三个场景、两个物理域和 seeds 48--50，共 126/126 个唯一实验单元。全部单元保留；每个单元均存在 resolved config、trajectory、metrics 与 provenance。规范化配置哈希、Git SHA、checkpoint 哈希和 manifest 哈希均通过复核，66 项相关回归测试通过。

关键 development 描述统计如下：

| 方法 | 成功 | 碰撞 | 平均终点距离 / m | 平均步数 |
|---|---:|---:|---:|---:|
| Simple Combination | 18/18 | 0/18 | 0.298212 | 426.56 |
| Value-only | 18/18 | 0/18 | 0.298213 | 426.22 |
| HSS-only | 17/18 | 0/18 | 0.296530 | 427.44 |
| Full Proposed | 18/18 | 0/18 | 0.298202 | 419.22 |

Development Gate 中“成功率方向有利”在成功率达到上限时按非负方向解释，即 Full 不低于 Simple；本轮二者均为 100%，因此这里只证明没有 development-level success regression，并不证明优越性。终点距离优势数值很小；平均步数降低约 7.33 步是待封存集检验的效率信号。上述结果只用于决定是否有资格开启确认实验，不作为正式论文确认结论。

## 3. 冻结方法与公共条件

七方法固定为：

1. Traditional MPPI；
2. ICODE-MPPI；
3. RL-driven MPPI；
4. Simple Combination；
5. Value-only；
6. HSS-only；
7. Full Proposed。

公共条件固定为：

- 三个场景：`lab_complex`、`narrow_corridor`、`u_trap_long_board`；
- 两个物理域：`nominal_seen`、`combined_unseen`；
- 每拍 100 次模型 rollout；RL 变体为两轮、每轮 50 条候选；
- 最大 600 control steps；
- pose/twist 使用带既有小噪声的 MuJoCo ground truth，以隔离 localization confound；
- LaserScan、scan_guard、local obstacle layer 与 safety arbitration 始终启用；
- planner 不读取全局障碍真值；
- Memory-Augmented MPPI 关闭；
- completion handover 关闭；
- role-aware HSS 固定为 `policy_rescue`，`policy_rescue_floor=0.50`；
- 不改变 Actor、ordinary ICODE ensemble、value-aligned ICODE ensemble 或 reliability calibration checkpoint。

## 4. 封存 seeds 与选择程序

正式独立 seeds 固定为：

```text
78006, 78007, 78008, 78009, 78010,
78011, 78012, 78013, 78014, 78015
```

选择过程不读取这些 seeds 的运行结果。程序扫描 `configs/`、`docs/` 与 `results/research_platform/rl/` 中的 progress、schedule、provenance 及相关配置记录；在候选区间 78001--78100 内，`78006--78015` 是扫描得到的第一个连续十 seed 未使用区间。选择标准仅为“连续且从未使用”，不含性能筛选。qualification seed 99、development seeds 48--50、L214 seeds 101--110 均不进入本实验。

`configs/research/complex_navigation_sealed_l217.yaml` 是唯一正式 manifest。正式 runner 必须从该 manifest 读取 seed；命令行不能替换 seed。若并行运行，只允许将上述固定列表按 `seed[index::count]` 确定性分片。完整随机调度由 `schedule_seed=20260724` 一次生成，分片保留其原始全局顺序。

## 5. 假设、比较与统计单位

独立统计单位为 seed；scene 与 physics domain 是 seed 内重复分层。禁止把 timestep、MPPI sample 或单个 scene-domain cell 当作独立重复。

主要比较为 `Full Proposed` 对 `Simple Combination`。按以下层级报告，不因结果改变顺序：

1. success rate 与 collision rate；
2. final goal distance 与 time-to-goal / executed steps；
3. trajectory length、minimum clearance、stuck/spin steps、mean absolute omega、control jerk；
4. planner mean/p95/max compute time；
5. HSS authority、guided fraction、terminal-value authority 等机制诊断量。

同时报告四核心臂的 $2\times2$ factorial：value alignment、role-aware adaptive HSS 的主效应与交互效应；七方法均报告描述统计。配对推断使用 seed-cluster bootstrap 10,000 次，场景和物理域先在 seed 内配对聚合。报告效应方向、点估计和 95% 区间，不把 development 调参或单个 seed 作为证据。

## 6. 冻结后的禁止事项

提交并推送本预注册后：

- 十个 sealed seeds 只运行一次完整矩阵；
- 不得按结果删除 seed、episode、碰撞或失败；
- 不得调阈值、换 checkpoint、改场景、改物理域或改成功半径后仍称同一确认实验；
- 运行异常只允许根据日志恢复缺失的预注册单元，且必须保留异常与恢复记录；
- sealed 结果若为负，必须如实保留并报告，不回到同一批 seed 调参；
- 不得把零碰撞解释为全局安全保证；
- 不得声称复现原始 ICODE 全部理论保证。

## 7. 完整性与交付要求

预期规模为 10 seeds × 3 scenes × 2 domains × 7 methods = 420 episodes、60 个随机完全区组。最终必须核对：

- 420/420 行及 420 个唯一键；
- 每个 block 恰有七方法；
- 全部 `qualification=0`；
- manifest、Git SHA、checkpoint 与 calibration 哈希；
- 每回合 resolved config、trajectory、metrics、provenance；
- 汇总 CSV、seed-cluster 统计 JSON、论文级图、中文报告及原始数据说明；
- 完整交付包复制到 Windows 桌面；
- 相关回归测试通过后提交并推送。
