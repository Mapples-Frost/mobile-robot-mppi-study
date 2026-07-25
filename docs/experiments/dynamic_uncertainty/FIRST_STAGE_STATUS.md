# First-Stage Dynamic-Uncertainty Status

Date: 2026-07-23  
Platform: native Windows  
Branch: `codex/change-aware-probabilistic-mppi`

## 本轮完成

- 完成 Windows 仓库、Git、最新实验、接口和测试基线审计；
- 将 Tracking 标记为 archived development line，保留完整正负结果；
- 冻结动态不确定性项目预注册、split 和 seed 规则；
- 实现四种单障碍物随机运动过程：
  - Noisy CV；
  - Speed Change；
  - Direction Change；
  - Combined Change + Occlusion；
- 分离过程噪声与观测噪声随机流；
- 实现 Deterministic CV；
- 实现 Gaussian CV Kalman（Joseph covariance update）；
- 实现遮挡期间 predict-only 更新；
- 实现 CSV/JSON/YAML 工件、SHA-256 provenance 和 integrity audit；
- 生成四过程障碍物轨迹图；
- 完成 4 processes x 2 noise settings x 5 development seeds = 40 runs；
- 没有运行 IMM、MPPI、RL、ICODE 或 HSS；
- 没有使用 sealed seed。

主要代码：

- `src/mobile_robot_mppi/obstacles/motion.py`
- `src/mobile_robot_mppi/obstacles/prediction.py`
- `src/mobile_robot_mppi/obstacles/artifacts.py`
- `src/mobile_robot_mppi/obstacles/visualization.py`
- `experiments/dynamic_uncertainty/run_probability_smoke.py`

主要协议与报告：

- `docs/experiments/dynamic_uncertainty/repository_audit.md`
- `docs/experiments/tracking/FINAL_DEVELOPMENT_STATUS.md`
- `docs/experiments/dynamic_uncertainty/PROGRAM_PREREGISTRATION.md`
- `docs/experiments/dynamic_uncertainty/PROBABILITY_NOTES_STAGE1.md`

工件：

- `research_artifacts/dynamic_uncertainty_probability_smoke/`
- 40 个 run 目录；
- 287 个文件；
- 约 12.26 MB；
- 完整性报告：`integrity_audit.json`；
- 障碍物图：`figures/obstacle_processes_smoke.png`。

## 核心数据

| Process | Noise | Deterministic ADE (m) | Kalman ADE (m) | Kalman 90% coverage |
|---|---:|---:|---:|---:|
| Noisy CV | low | 0.3075 | 0.0340 | 0.934 |
| Noisy CV | medium | 0.9093 | 0.0830 | 0.937 |
| Speed Change | low | 0.3338 | 0.1533 | 0.392 |
| Speed Change | medium | 0.9777 | 0.1924 | 0.474 |
| Direction Change | low | 0.3291 | 0.3037 | 0.402 |
| Direction Change | medium | 0.9224 | 0.3313 | 0.476 |
| Combined Change | low | 0.3347 | 0.3850 | 0.300 |
| Combined Change | medium | 0.9244 | 0.4193 | 0.307 |

总体 smoke 描述值：

- Deterministic CV mean ADE: 0.6299 m；
- Gaussian CV Kalman mean ADE: 0.2377 m；
- Gaussian CV Kalman 90% coverage: 0.5278；
- minimum covariance eigenvalue: positive in every run；
- reproducible trajectories: 40/40；
- duplicate experimental keys: 0；
- missing required artifacts: 0。

这些数值只用于工程诊断。Smoke 样本和参数不是 Predictor Gate，不支持
正式方法优越性结论。

## 影响解释

Kalman 在 Noisy CV 中的 90% coverage 约为 93%--94%，说明线性高斯模型在
运动规律正确时能够给出接近合理的置信范围。发生速度或方向突变后，
coverage 下降到约 30%--48%。这表示单一 CV 模型在突变后过度自信：
协方差矩阵虽然数值合法，但没有覆盖真实运动模式变化。

行为层面的含义是，如果现在直接把这个概率送入风险规划器，控制器可能把
危险候选误判为低风险。因此当前剩余安全余量不可定义，概率风险阈值仍被
禁止用于闭环控制。

## 测试结果

新增动态不确定性测试：

```text
17 passed
```

覆盖：

- 四类运动过程；
- 噪声源分离；
- seed 复现；
- Kalman 状态更新；
- 遮挡 predict-only；
- covariance finite/symmetric/PSD；
- no future truth leakage；
- 40-run artifact integrity。

仓库改动前全测试基线在补齐已声明的 Matplotlib 依赖后为：

```text
977 passed, 5 failed
```

五个失败均为历史 L34/L70/L72 checkpoint 文件缺失。它们未被修改或隐藏。

加入本阶段代码后的全套回归为：

```text
994 passed, 5 failed
```

失败集合与改动前完全相同；通过数增加的 17 项全部来自本阶段测试。

## Gate

Engineering smoke: **PASS**

通过项：

- 40/40 runs 完整；
- 所有真值和可用观测有限；
- CV/Kalman prediction 有限；
- 所有 Kalman covariance 对称且 PSD；
- 40/40 轨迹可复现；
- 无重复 key；
- 无 sealed seed；
- 工件结构完整。

尚未执行：

- Predictor Gate；
- probability calibration Gate；
- collision-risk Gate；
- closed-loop planning Gate。

## 下一步

唯一推荐动作是实现普通 IMM：

```text
CV + Turn-L + Turn-R + Brake/Stop
```

先只实现模式混合、各模型 Kalman 更新、观测 likelihood、模式概率归一化和
未来 mixture 输出；暂不加入 NIS change detector。普通 IMM 的数值不变量和
held-out-free development smoke 通过后，再单独加入 change awareness，才能
识别提升来自多模型表示还是突变检测机制。

本阶段到此暂停，等待人工审阅。
