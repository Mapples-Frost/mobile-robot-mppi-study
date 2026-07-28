# AR(1) Chapter 1 MuJoCo 闭环 A/B 筛选审计

完成时间：`2026-07-28T01:34:40-04:00`

最终 Gate：`PASS_TO_CONFIRMATORY`

> 这是 12 对 seed 的筛选实验，不是统计确认实验。两组均为 0 次成功；Gate 仅由预冻结的配对最终目标距离门槛触发。

## 1. 研究问题

在保持 Actor、ICODE、CA-IMM/预测器、风险阈值、安全边距、代价函数、安全仲裁、地图、rollout 数量、episode 上限、seed 和停止条件不变时，将 MPPI 的高斯候选从逐时间步独立 `iid` 改为 `ar1:2.0`，离线候选覆盖率优势能否转化为 Chapter 1 的闭环改善。

本轮未引入任何新机制，也未改动重要性采样协方差修正、rate-limit 处理或 Actor 权重。

## 2. 代码版本与 Git 状态

- 仓库：`D:\Projects\mobile-robot-mppi-study-single-v6`
- 分支：`codex/complex-static-three-dynamic-v1`
- 正式运行前提交：`424a18380f02d5485a8376145c699565ce225c4a`
- 正式运行前 tag：`ar1-noise-basis-pre-execution`
- 运行前 tracked worktree：已提交；历史遗留的 untracked 日志、研究产物和脚本均未删除、覆盖或纳入本次代码提交。
- 正式结果原先不存在；本次产物使用独立 seed/arm 目录并由 runner 拒绝覆盖。

## 3. 协议、哈希与唯一差异

- 协议：`configs/research/noise_basis_ab_development_v1.yaml`
- 协议 SHA256：`f1dfd4098941bb9101c0efc525295ac31df3c49a6aa7bba79fa2d5c37cc68d0e`
- 地图：Chapter 1
- 每次规划预算：600 rollouts（300 candidates × 2 iterations）
- horizon：36
- episode 上限：1200 steps
- 控制周期：0.1 s
- optimizer：`paper_rl_driven`

两臂解析配置逐字段比较后，除非行为性 arm 元数据外，唯一行为差异为：

```yaml
planner:
  noise_basis: "iid"
```

与：

```yaml
planner:
  noise_basis: "ar1:2.0"
```

## 4. Seed 与 common random numbers

配对 seed：

`791101201, 791101202, 791101203, 791101204, 791101205, 791101206, 791101207, 791101208, 791101209, 791101210, 791101211, 791101212`

运行顺序为每个 seed 先 iid 后 AR(1)，然后进入下一个 seed。相同 seed 使用相同地图、障碍物轨迹、初始状态、Actor/模型、风险配置、rollout 预算和停止规则。预执行扫描未发现这些 seed 已用于既有 episode 产物。

## 5. 运行环境

- Windows 10 build 26200，PowerShell 5.1.26100.8875
- Python 3.10.11（仓库 `.venv`）
- MuJoCo 3.2.3
- PyTorch 2.13.0+cpu
- NumPy 1.26.4
- pytest 9.1.1
- 项目包可由该虚拟环境正常导入
- 本次 planner 使用 CPU；正式开始前确认没有其他项目 Python/MuJoCo episode 在运行

## 6. 实现审计与预执行修正

静态检查确认 `MppiConfig.noise_basis`、映射读取、合法/非法格式验证、默认 iid 分支、basis 构造器以及回归测试均存在。

正式运行前发现一项会使原 A/B 无效的问题：Chapter 1 实际解析到 `PaperRLDrivenMppiController`，其真实高斯候选入口 `_gaussian_samples` 仍直接调用 iid `rng.normal`，没有经过先前只接入 `MppiController._sample` 的 basis 分支。因此原 runner 虽会写出不同 arm 标签，两臂真实候选却都会是 iid。

在 0 个正式 episode 的时点修复并重新冻结：

1. 将 RL-driven 与 Paper-RL 的真实 Gaussian population 接入 `planner.noise_basis`。
2. iid 分支保留原始 `rng.normal` 表达式和随机数消费顺序。
3. AR(1) 只作用于标准化 Gaussian perturbation，随后乘既有逐步标准差。
4. 增加只读 rate-limit 遥测。
5. 将三个 Gate 阈值写入协议，并让 runner/analyzer 从协议读取。
6. 删除协议中不可达的 `INCONCLUSIVE` 文本分支；阈值本身未改变。

问题和修正详见 `research_artifacts/ar1_closed_loop_screen_ch1/PRE_EXECUTION_AMENDMENT.md`；冻结文件哈希详见 `PRE_EXECUTION_MANIFEST.json`。

## 7. 回归测试

- 专用 noise-basis suite：`33 passed in 0.89s`
- noise-basis 加 RL/Paper-RL planner suite：`71 passed in 0.87s`
- 初始接管时的原专用 suite：`27 passed`

已验证：

- 默认 iid 与逐字历史实现 `np.array_equal`
- 相同 seed 下后续 RNG 流一致
- 完整 `MppiController._sample` 输出逐比特一致
- `samples[0] = prior.mean` 不变
- bounds clipping 不变
- 默认配置解析为 `"iid"`
- 实际 Paper-RL Gaussian 路径的输出和 RNG continuation bit-exact
- AR(1) 在实际 Paper-RL 路径产生预期时间相关性

## 8. 单 episode smoke test

手动运行 `seed=791101201, arm=control, noise_basis=iid`，未直接启动全部 24 个 episode。

- MuJoCo 场景成功加载
- resolved config 为 Chapter 1、`paper_rl_driven`、600 rollouts、`noise_basis: iid`
- runner 完整运行并产生 403 行、618 列 trajectory
- outcome：`collision`
- final goal distance：6.586869 m
- planner、候选可行率、风险、安全仲裁、Actor/reliability、动作和 rate-limit 字段均存在
- `metrics.json`、`trajectory.csv`、resolved config/protocol、各自 SHA256 和 provenance 均完整
- episode stderr 为空

外层交互 shell 在 10 秒时超时，但 episode 子进程继续正常运行至完成；其产物未删除或重跑。随后正式 `-Resume` 检测到该完整 control 产物并从配对 treatment 继续。

## 9. 产物完整性

所有 24 个 episode 均完成；每个目录均包含：

`config_resolved.yaml`, `config_sha256.txt`, `protocol_resolved.yaml`, `protocol_sha256.txt`, `provenance.json`, `trajectory.csv`, `metrics.json`

所有 episode 的协议哈希均与冻结 SHA256 一致，arm 配置均匹配，未发现缺失文件或非空 per-episode stderr。batch stderr 为空。

| seed | iid 产物 | AR(1) 产物 |
|---:|---|---|
| 791101201 | `research_artifacts/noise_basis_ab_v1/seed791101201/control` | `research_artifacts/noise_basis_ab_v1/seed791101201/treatment` |
| 791101202 | `research_artifacts/noise_basis_ab_v1/seed791101202/control` | `research_artifacts/noise_basis_ab_v1/seed791101202/treatment` |
| 791101203 | `research_artifacts/noise_basis_ab_v1/seed791101203/control` | `research_artifacts/noise_basis_ab_v1/seed791101203/treatment` |
| 791101204 | `research_artifacts/noise_basis_ab_v1/seed791101204/control` | `research_artifacts/noise_basis_ab_v1/seed791101204/treatment` |
| 791101205 | `research_artifacts/noise_basis_ab_v1/seed791101205/control` | `research_artifacts/noise_basis_ab_v1/seed791101205/treatment` |
| 791101206 | `research_artifacts/noise_basis_ab_v1/seed791101206/control` | `research_artifacts/noise_basis_ab_v1/seed791101206/treatment` |
| 791101207 | `research_artifacts/noise_basis_ab_v1/seed791101207/control` | `research_artifacts/noise_basis_ab_v1/seed791101207/treatment` |
| 791101208 | `research_artifacts/noise_basis_ab_v1/seed791101208/control` | `research_artifacts/noise_basis_ab_v1/seed791101208/treatment` |
| 791101209 | `research_artifacts/noise_basis_ab_v1/seed791101209/control` | `research_artifacts/noise_basis_ab_v1/seed791101209/treatment` |
| 791101210 | `research_artifacts/noise_basis_ab_v1/seed791101210/control` | `research_artifacts/noise_basis_ab_v1/seed791101210/treatment` |
| 791101211 | `research_artifacts/noise_basis_ab_v1/seed791101211/control` | `research_artifacts/noise_basis_ab_v1/seed791101211/treatment` |
| 791101212 | `research_artifacts/noise_basis_ab_v1/seed791101212/control` | `research_artifacts/noise_basis_ab_v1/seed791101212/treatment` |

机器可读汇总为 `research_artifacts/noise_basis_ab_v1/ab_screen_result.json`，配对表为 `paired_results.csv`。

## 10. 字段映射

| 要求概念 | 实际字段/计算 |
|---|---|
| outcome / stop reason | `metrics.json: outcome`, `termination_reason` |
| success / collision / timeout | outcome 派生布尔值 |
| completion steps | `steps` |
| final goal distance | `final_goal_distance` |
| path progress | `path_progress_m`, `path_progress_ratio` |
| minimum static clearance | `minimum_static_clearance` |
| minimum dynamic distance | `minimum_dynamic_obstacle_center_distance`（中心距离，不冒充表面 clearance） |
| minimum overall clearance | `minimum_overall_clearance` |
| zero-feasible | `zero_feasible_step_count`, `zero_feasible_run_count`, `longest_zero_feasible_run` |
| hard violation | `hard_violation_count`, `hard_violation_frequency` |
| fallback | `fallback_activation_count`, `fallback_activation_frequency` |
| candidate feasible | `candidate_feasible_fraction_mean/min` |
| planner timing | `planner_compute_ms_mean/p50/p95/max`, `planner_deadline_miss_rate` |
| stuck | `stuck_steps` |
| applied velocity | `applied_v_*`, `applied_omega_*`, `applied_action_delta_l2_*` |
| Actor/guided | `gaussian_candidate_fraction`, `guided_candidate_fraction`, elite survival、reliability 字段 |
| rate-limit | `paper_gaussian_*rate_limit*` 和 trajectory 相邻动作复算 |

## 11. 12 对结果

表中的 clearance 为 minimum overall clearance，P95 单位为 ms。

| seed | iid outcome | ar1 outcome | iid final m | ar1 final m | iid clearance m | ar1 clearance m | iid steps | ar1 steps | iid P95 | ar1 P95 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 791101201 | collision | collision | 6.587 | 6.612 | -0.001 | -0.016 | 403 | 893 | 974.97 | 909.72 |
| 791101202 | collision | collision | 5.227 | 5.312 | -0.010 | -0.001 | 591 | 89 | 359.04 | 357.74 |
| 791101203 | collision | collision | 5.995 | 6.102 | -0.015 | -0.005 | 656 | 73 | 362.08 | 349.18 |
| 791101204 | collision | timeout | 6.055 | 2.316 | -0.020 | 0.136 | 86 | 1200 | 378.20 | 339.72 |
| 791101205 | timeout | timeout | 4.668 | 2.411 | 0.156 | 0.112 | 1200 | 1200 | 339.81 | 831.69 |
| 791101206 | timeout | collision | 4.039 | 5.256 | 0.097 | -0.000 | 1200 | 90 | 330.65 | 908.61 |
| 791101207 | collision | collision | 5.480 | 5.273 | -0.025 | -0.007 | 437 | 87 | 933.85 | 347.68 |
| 791101208 | collision | timeout | 6.027 | 2.315 | -0.005 | 0.116 | 195 | 1200 | 926.54 | 859.28 |
| 791101209 | collision | timeout | 5.290 | 2.367 | -0.004 | 0.123 | 209 | 1200 | 1114.76 | 807.08 |
| 791101210 | timeout | timeout | 2.762 | 2.403 | 0.003 | 0.101 | 1200 | 1200 | 794.24 | 335.60 |
| 791101211 | collision | timeout | 5.375 | 2.423 | -0.008 | 0.094 | 203 | 1200 | 353.29 | 334.54 |
| 791101212 | collision | collision | 5.408 | 5.468 | -0.028 | -0.002 | 449 | 201 | 346.97 | 362.56 |

## 12. 聚合结果

| 指标 | iid | AR(1) |
|---|---:|---:|
| success | 0 | 0 |
| collision | 9 | 6 |
| timeout | 3 | 6 |
| final distance，episode median (m) | 5.392 | 3.840 |
| steps，episode median | 443 | 1046.5 |
| minimum overall clearance，episode median (m) | -0.0066 | 0.0469 |
| zero-feasible steps，总计 | 230 | 92 |
| zero-feasible runs，总计 | 51 | 17 |
| longest zero-feasible run，episode median | 7.5 | 3 |
| hard violations，总计 | 443 | 187 |
| fallback activations，总计 | 1240 | 645 |
| candidate feasible fraction，episode mean | 0.8773 | 0.8969 |
| stuck steps，总计 | 1686 | 1890 |
| stuck step rate（总计/总 steps） | 24.69% | 21.89% |
| planner mean latency，episode mean (ms) | 620.42 | 378.60 |
| planner P95，episode mean (ms) | 601.20 | 561.95 |
| deadline miss rate，episode mean | 94.10% | 98.88% |
| applied `|v|`，episode mean | 0.2417 | 0.2960 |
| applied `|omega|`，episode mean | 0.2626 | 0.2943 |
| applied action delta L2，episode mean | 0.1160 | 0.1101 |

配对统计：

- 净成功对数：0
- treatment-only collision：1 对（791101206）
- control-only collision：4 对（791101204、208、209、211）
- 配对最终目标距离改善中位数：0.283041 m
- treatment-control steps 中位数：0
- treatment-control minimum static clearance 中位数：-0.017134 m
- treatment-control minimum overall clearance 中位数：+0.022030 m
- treatment-control zero-feasible steps 中位数：-15
- treatment-control stuck steps 中位数：+14.5
- treatment-control planner P95 中位数：-28.610 ms

AR(1) 没有提高成功数，但把 4 个 iid 碰撞转为 timeout，同时把 1 个 iid timeout 转为碰撞。它改善了目标距离、整体 clearance、zero-feasible、hard violation 和 fallback；总 stuck steps 因更多完整 timeout 而增加，但按总步数归一化略降。它并非单纯延长所有运行：6 个 treatment 达到上限、6 个提前碰撞；配对 steps 差的中位数为 0。

## 13. 动作变化率

动作空间 rate limit 为 `[0.6, 2.0]` 每秒，`dt=0.1`，即每步 `[0.06, 0.2]`。

| 指标 | iid | AR(1) |
|---|---:|---:|
| raw Gaussian 元素违反比例，episode mean | 53.66% | 9.48% |
| raw Gaussian 候选至少一次违反，episode mean | 84.00% | 64.80% |
| 既有限幅后 Gaussian 元素违反比例 | 0 | 0 |
| proposed command 相邻步违反数 | 657 | 364 |
| executed command 相邻步违反数 | 813 | 787 |
| applied interval-average 相邻行诊断计数 | 999 | 771 |

executed violation 归因：

| 类别 | iid | AR(1) |
|---|---:|---:|
| safety override | 261 | 553 |
| emergency candidate selected | 480 | 220 |
| active fallback | 11 | 0 |
| regular/weighted | 61 | 14 |

因此，候选 Gaussian limiter 在两臂均正确工作。最终命令的超限大多对应既有 emergency candidate（代码明确保护其不受首动作 slew clipping）或 safety override；普通/加权路径仍有少量实测超限。本轮未修改 rate-limit 逻辑。完整定义和计数见 `RATE_LIMIT_AUDIT.json`。

## 14. Actor、guided 与安全模块

| 指标，episode mean | iid | AR(1) |
|---|---:|---:|
| Gaussian candidate fraction | 99.8869% | 99.8201% |
| guided candidate fraction | 0.1131% | 0.1799% |
| Gaussian elite survival | 16.2218% | 16.2677% |
| guided elite survival | 0.7022% | 0% |
| reliability authority | 0.1591 | 0.3508 |
| safety override steps，总计 | 1620 | 3811 |

supervised maneuver Actor candidate fraction 和 selected steps 在两臂均为 0。这里的执行动作是 MPPI 加权更新，不能诚实地标为单一 Gaussian 或 guided 来源；实际 guided 份额很小。AR(1) 改变了闭环状态访问分布，使 reliability authority 和 safety override 频次明显变化，但没有改动 Actor 或可靠性逻辑本身。AR(1) 的 guided elite survival 没有上升。

## 15. 重要性采样协方差限制

`_sampling_covariance` 在 generic standard MPPI 的可选 importance correction 中提供逐步边际对角协方差；若对相关序列启用该 correction，跨时间协方差未建模，会形成“整条序列独立”的近似。

本次实际路径为 Paper-RL：

- `importance_sampling_correction: false`
- Paper-RL 显式拒绝开启该 correction
- `paper_standard_fallback_on_advantage_veto: false`
- 所有 episode 的 `paper_standard_fallback_active_fraction` 最大值为 0

因此该边际协方差近似没有进入本次 A/B 的实际权重计算，不构成本次组间混杂。但它仍是未来若启用相关采样 importance correction 时必须单独消融的已知近似；本轮未修改。

## 16. 预冻结 Gate 与最终判定

按安全优先顺序：

1. `FAIL_SAFETY`：要求至少 2 对 treatment-only collision；实测 1 对，未触发。
2. `PASS_TO_CONFIRMATORY`：要求净成功至少 +2，或配对最终目标距离改善中位数至少 0.25 m；实测净成功 0，但距离改善 0.283041 m，触发。
3. 因已触发第二条，不判 `FAIL_NO_EFFECT`。

最终判定：`PASS_TO_CONFIRMATORY`

门槛余量只有约 0.033 m，而且两臂均无成功。这是“值得做 fresh-seed 确认”的筛选信号，不是 AR(1) 已被证实提高闭环成功率。

## 17. 失败、异常与证据保留

- 修复前发现 Paper-RL 高斯入口绕过 noise basis；0 个正式 episode 时停止，记录问题，修复、重测、重新哈希和提交后才运行。
- 预执行发现一个历史临时目录中的无限步 MuJoCo viewer，以及超时的只读 seed-scan 进程；按确切 PID 终止，未删除其输出。
- smoke 外层 shell 超时但 episode 正常完成；保留空 stdout/stderr 和完整 episode 产物，正式 runner 用 `-Resume` 跳过。
- 24 个 episode 全部完成，所有 stderr 为空；未为改善结果更换 seed、地图、阈值或停止标准。
- 旧的 `research_artifacts/noise_basis_ab_v1/AUDIT_RECORD.md` 是此前 Linux 环境下的“PREPARED, NOT RUN”历史记录，未覆盖或删除；本报告及实际产物 supersede 其执行状态，但保留其历史证据属性。

## 18. 局限性与下一步

局限性：

- 只有 12 对 seed，未做显著性推断，不能作为最终统计结论。
- 两组都没有成功，screen Gate 由一个接近门槛的连续代理指标触发。
- AR(1) 将更多碰撞变成 timeout，但仍未完成任务；“更接近目标”不等于成功。
- planner deadline miss 在两臂都近乎饱和，且仅 CPU 环境；延迟均值受少数 25–63 秒极端值影响。
- treatment 访问了不同闭环状态，Actor reliability、安全 override、运行长度和遥测分母随之变化，不能将所有下游差异解释为直接采样效应。
- minimum dynamic 指标是障碍物中心距离，不是表面 clearance。

由于 Gate 为 `PASS_TO_CONFIRMATORY`，下一步仅建议：另行预注册 fresh seeds、至少 20 对的确认实验，保持本次所有行为配置和安全优先 Gate 不变；在确认实验中将成功率/碰撞率作为主要结果，把目标距离作为次要连续结果。不得复用本次 12 个 seed，也不得根据本次结果调整阈值。

建议结果提交与 tag：

```powershell
git add research_artifacts/noise_basis_ab_v1 `
        research_artifacts/ar1_closed_loop_screen_ch1/AUDIT.md `
        research_artifacts/ar1_closed_loop_screen_ch1/RATE_LIMIT_AUDIT.json
git commit -m "experiment: record AR1 Chapter 1 closed-loop screen"
git tag -a ar1-noise-basis-screen-result -m "AR1 Chapter 1 screen result"
```
