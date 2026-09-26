# L19：跨训练种子 critic advantage margin 校准结果与数据审计

日期：2026-07-15  
性质：calibration 阶段负结果；selection 未开启  
预注册：[`18_l19_advantage_margin_calibration_prereg_2026-07-15.md`](18_l19_advantage_margin_calibration_prereg_2026-07-15.md)

## 1. 结论先行

L19 数据质量 Gate 通过，但六个预注册 target-critic margins 均未通过全局 fail-closed 选择条件：

- calibration 使用 3 个独立 training seeds、每个 12 个全新 episode seeds；
- BC pooled success 为 `29/36`，所有条件均为 `0/36` collision；
- 表现最好的 `margin=0.02` 达到 `31/36`，pooled mean return delta 为 `+1.620`，mean goal-distance improvement 为 `+0.0367 m`；
- 但它在 training seed `20260722` 上损失 3 个 BC-success episodes，mean return delta 为 `-3.013`；
- 其他 margin 也都存在 paired BC-success loss，并至少在一个 training seed 上平均回报下降；
- selector 因而冻结 `bc_fallback`，`selected_margin=null`；
- 预留 selection seeds `20281101--20281115` 没有运行。

这说明：一个固定的 raw Q-difference margin 不能稳定跨越当前三个 independently trained critics。总体均值上的改善不能抵消某个训练模型上的明显退化。

## 2. 实际实验设计

固定 target-critic gate：

\[
a(s)=
\begin{cases}
a_{\mathrm{corr}}, &
\min_i[Q_i^{\mathrm{target}}(s,a_{\mathrm{corr}})
-Q_i^{\mathrm{target}}(s,a_{\mathrm{BC}})]\ge m,\\
a_{\mathrm{BC}}, & \text{otherwise}.
\end{cases}
\]

候选 margin：

```text
0.000, 0.005, 0.010, 0.020, 0.040, 0.080
```

实验保持静态 `u_trap_long_board`、nominal prediction、Memory 关闭、ICODE 关闭。三个训练 checkpoint 使用同一个 margin 网格和相同 calibration episode seeds `20281001--20281012`。BC 在每个 training seed/episode seed 上只计算一次。

## 3. Calibration 结果

| target margin | success | BC-success losses | success gains | collision | mean return delta | mean distance improvement | accept fraction | eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BC | 29/36 | — | — | 0/36 | — | — | — | reference |
| 0.000 | 26/36 | 5 | 2 | 0/36 | -2.832 | -0.0797 m | 0.761 | no |
| 0.005 | 29/36 | 5 | 5 | 0/36 | -0.194 | -0.0385 m | 0.677 | no |
| 0.010 | 24/36 | 7 | 2 | 0/36 | -4.816 | -0.1315 m | 0.587 | no |
| 0.020 | **31/36** | **3** | 5 | 0/36 | **+1.620** | **+0.0367 m** | 0.493 | **no** |
| 0.040 | 28/36 | 5 | 4 | 0/36 | -1.105 | -0.0589 m | 0.294 | no |
| 0.080 | 30/36 | 4 | 5 | 0/36 | +0.691 | +0.0011 m | 0.117 | no |

`success = BC successes - paired losses + paired gains`。margin `0.02` 的净 success 虽然增加 2，但仍把 3 个原本成功的配对任务变成失败，违反预注册 non-inferiority 条件。

### 3.1 margin 0.02 的跨训练种子差异

| training seed | BC-success losses | success gains | mean return delta | mean distance improvement | accept fraction |
|---:|---:|---:|---:|---:|---:|
| 20260721 | 0 | 1 | +2.686 | +0.0750 m | 0.526 |
| 20260722 | **3** | 2 | **-3.013** | -0.0278 m | 0.376 |
| 20260723 | 0 | 2 | +5.186 | +0.0628 m | 0.577 |

如果只报告 pooled success 或只挑 seed `20260721/20260723`，会得到“margin 有效”的错误印象。training seed `20260722` 是必要的独立反例。

## 4. 预注册 selector 判定

| margin | no success loss | no collision | every-seed return noninferiority | nondegenerate | meaningful benefit | selected |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | fail | pass | fail | pass | pass | no |
| 0.005 | fail | pass | fail | pass | pass | no |
| 0.010 | fail | pass | fail | pass | pass | no |
| 0.020 | fail | pass | fail | pass | pass | no |
| 0.040 | fail | pass | fail | pass | pass | no |
| 0.080 | fail | pass | fail | pass | pass | no |

最终决策：

```json
{
  "selected_mode": "bc_fallback",
  "selected_margin": null,
  "selection_reason": "no_preregistered_margin_eligible",
  "selection_seeds_opened": false
}
```

## 5. CSV/JSON 数据审计

### 5.1 文件与格式

每个 training seed 目录包含：

- `config_snapshot.json`：约 14 KB；
- `summary.json`：约 6 KB；
- `episodes.csv`：84 行，约 43 KB；
- `paired_episodes.csv`：72 行，约 48 KB；
- `steps.csv`：约 16.6--17.2 MB。

CSV 为 UTF-8 明文表格，JSON 保存配置、checkpoint provenance、完整 margin mapping 和 selector 决策。

### 5.2 结构与完整性

| training seed | episode rows | paired rows | control-step rows | episode duplicate keys | paired duplicate keys | step duplicate keys |
|---:|---:|---:|---:|---:|---:|---:|
| 20260721 | 84 | 72 | 23,483 | 0 | 0 | 0 |
| 20260722 | 84 | 72 | 24,275 | 0 | 0 | 0 |
| 20260723 | 84 | 72 | 23,439 | 0 | 0 | 0 |
| total | 252 | 216 | **71,197** | 0 | 0 | 0 |

质量检查全部通过：

- 每个 training seed 均完整包含 `12 × (1 BC + 6 margins) = 84` episodes；
- condition、episode seed、step 复合键无重复；
- 关键 advantage、gate、correction、return 和距离字段无缺失；
- 所有数值字段无 NaN/Inf；
- BC raw correction 最大值精确为 `0.0`；
- 三个 run 的 margin 顺序、condition mapping 和 calibration seeds 完全一致；
- 所有 checkpoint 的 frozen BC hash 在各自 baseline/candidate 对内一致。

### 5.3 异常值解释

return 呈现明显的成功/失败双峰：到达目标的 episode 通常为正回报，跑满步数的失败 episode 会产生较大负回报。这不是缺失值或数值异常，而是任务终止结果的结构。分析因此优先使用 paired success loss，再使用 return 和距离，而不是只比较均值。

## 6. 工程验证

新增实现：

- `src/mobile_robot_mppi/rl/calibration.py`：可测试、fail-closed 的全局 margin selector；
- `experiments/rl/run_correction_advantage_margin_grid.py`：BC 单次运行、多 margin 闭环网格；
- `experiments/rl/select_correction_advantage_margin.py`：跨训练种子审计与冻结决策；
- `tests/rl/test_advantage_margin_calibration.py`：网格、排序、fallback、seed drift 与配置错误测试。

selector 会拒绝：不完整网格、重复 training seed、calibration seed 漂移、condition mapping 漂移、非有限数值、CSV 重复键或 BC correction 非零。

## 7. 科研解释

L19 排除了一个重要但过于简单的方案：

> “给所有训练出的 critics 统一加一个 raw Q-difference 阈值，就能可靠保护 BC。”

失败原因不是 margin 完全没作用。margin `0.02` 对两个模型有效，却对第三个模型有害。这表明不同训练 seed 的 critic 数值尺度或校准关系并不一致；raw Q difference 不能直接当作跨模型统一置信度。

当前证据支持下一步研究以下二选一，而不是继续扩大本轮网格：

1. **尺度归一化的 advantage**：例如相对 Q 尺度、回报尺度或 calibration quantile 的无量纲 score；
2. **独立风险 critic / ensemble**：不使用被 actor 直接优化的同一 critic 作为部署裁判。

若继续，应重新预注册并使用新 calibration 数据。不能在本批数据上添加 `0.015、0.025` 等阈值后声称确认性成功。

## 8. 结果位置

```text
results/research_platform/rl/
  l19_margin_calibration_seed20260721_20281001_20281012_20260715_v1/
  l19_margin_calibration_seed20260722_20281001_20281012_20260715_v1/
  l19_margin_calibration_seed20260723_20281001_20281012_20260715_v1/
  l19_margin_calibration_multiseed_20260715_v1/
    selection.json
    paired_calibration_rows.csv
```

## 9. 复现命令

单 training seed margin grid：

```bash
.venv/bin/python experiments/rl/run_correction_advantage_margin_grid.py \
  --config configs/rl/sac_mppi_utrap_advantage_margin_l19.yaml \
  --scene-config configs/research/mujoco_u_trap_long_board.yaml \
  --baseline-checkpoint RUN/checkpoints/initial.pt \
  --candidate-checkpoint RUN/checkpoints/step_000020000.pt \
  --training-seed 20260721 \
  --seeds 20281001,20281002,20281003,20281004,20281005,20281006,20281007,20281008,20281009,20281010,20281011,20281012 \
  --margins 0,0.005,0.01,0.02,0.04,0.08 \
  --output-dir results/research_platform/rl/L19_RUN
```

跨训练种子冻结选择：

```bash
.venv/bin/python experiments/rl/select_correction_advantage_margin.py \
  --run-dir L19_SEED_1 \
  --run-dir L19_SEED_2 \
  --run-dir L19_SEED_3 \
  --output-dir results/research_platform/rl/l19_margin_calibration_multiseed_20260715_v1
```
