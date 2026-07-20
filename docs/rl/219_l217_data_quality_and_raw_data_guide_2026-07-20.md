# L217 数据质量与原始数据说明

## 1. 数据位置

```text
results/research_platform/rl/complex_navigation_sealed_l217/
├── shard_0_of_3/
├── shard_1_of_3/
├── shard_2_of_3/
└── merged_analysis/
```

每个 shard 保存不可替换的原始回合；`merged_analysis` 只保存合并索引、统计表和图，不复制 raw runs。

## 2. 完整性审计

`merged_analysis/integrity_audit.json` 是机器可读的主审计。结果为：420 个回合、420 个唯一单元、60 个完整 block、0 重复、0 qualification、420 个 run artifact 集合通过、9 个外部哈希引用通过。

`progress.csv` 有 420 行。以下空值是设计预期，不是数据丢失：

- `cross_track_*`：420 行为空，因为本实验是 point-goal 而非 polyline path-tracking；
- `minimum_dynamic_obstacle_center_distance`：420 行为空，因为 L217 没有动态障碍；
- `time_to_goal_s`：121 行为空，恰好对应 121 个 `max_steps` 终止回合；
- checkpoint/calibration path 字段只对相应方法有意义，因此有结构性空值。

必需数值字段 `steps/success/collision/final_goal_distance/trajectory_length/minimum_clearance/control_jerk/stuck_steps/planner_compute/paper_total_rollouts` 均无 NaN/Inf/空值。

## 3. 原始回合目录

每个回合路径为：

```text
shard_i_of_3/runs/<method>/<method>__<scene>__<domain>__seed<seed>/
```

文件作用：

- `config_resolved.yaml`：include、物理域和方法 override 全部解析后的最终配置；
- `trajectory.csv`：逐控制拍状态、控制、安全仲裁、规划器和机制诊断；
- `metrics.json`：单回合聚合指标；
- `provenance.json`：该回合的 Git SHA 与规范化配置哈希。

## 4. 合并和统计产物

- `merged_analysis/progress.csv`：按冻结全局 run order 合并的 420 行主表；
- `schedule.json`：预先生成的区组内随机顺序；
- `provenance.json`：合并 provenance 与三个 source shard 哈希；
- `run_artifact_index.json`：420 个 run 路径、config hash 和 trajectory 行数；
- `analysis_audit.json`：统计单位、bootstrap 次数和分析输入哈希；
- `paired_comparisons.json`：全部预先声明 outcome 的配对 seed-cluster 结果；
- `factorial_contrasts.json`：核心 2×2 factorial 结果；
- `tables/*.csv`：用于论文/人工检查的扁平表；
- `figures/*.png|pdf`：300 dpi PNG 和矢量 PDF。

## 5. 统计解释

seed 是独立单位。同一 seed 的三个 scene 和两个 physics domain 是重复分层，因此置信区间在 seed 层面重采样，不能把 60 个 cell 当作 60 个独立样本。`steps` 是包含失败回合的 bounded completion-time endpoint；失败回合保留 600 步，避免只在成功子集计算 time-to-goal 造成选择偏差。

## 6. 质量限制

1. 所有障碍均为静态；
2. 三个地图的起点和终点方向相似；
3. 三个 shard 同机并发，绝对 wall-clock planner time 受资源争用影响；
4. 零碰撞只适用于这些 seeds、场景和安全配置，不是全局保证；
5. residual policy context 与 residual policy authority 的激活比例为零；
6. terminal-value authority 始终为 1，没有实现 reliability weighting；
7. L217 不能支持完整耦合方法的全部主张。

## 7. 防止误用

- 不得只保留成功回合；
- 不得从 sealed seeds 中挑选“代表性好看轨迹”作为统计证据；
- 图 4 固定使用封存列表第一个 seed 78006，而不是按效果选 seed；
- 不得把 episode、timestep 或 MPPI candidate 当作独立重复；
- 不得把 L217 后续重跑仍称为同一次确认实验；
- 如补做完整机制，必须新建 manifest、全新 sealed seeds 和新的实验编号。
