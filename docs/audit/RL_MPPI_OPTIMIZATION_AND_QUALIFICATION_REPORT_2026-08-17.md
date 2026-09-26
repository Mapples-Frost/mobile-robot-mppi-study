# RL-MPPI 优化与资格验证阶段报告

**报告日期：** 2026-08-17  
**项目：** Learning-Augmented Risk-Aware MPPI with Change-Aware Obstacle Forecasting  
**报告性质：** 阶段性工程/算法优化事实记录；不把审计改动表述为运动性能提升。  
**当前最高优先级：** 保留本报告所列已完成改动，停止继续扩张 replay/capsule/worktree/CUDA-graph 审计细节，转入实车失败定位或论文框架内的算法优化。

---

## 1. 执行摘要

本阶段完成的主要工作是让 RL-MPPI 的**离线重放、状态恢复、证据记录与失败隔离**变得可审计；它消除了若干会污染实车问题归因和算法调参的工程黑箱，但**尚未实施针对车辆运动性能的 MPPI 代价、采样、运动模式、Actor/HSS 权限或 residual-shield 策略优化**。

最重要的技术发现是：当前 residual CUDA-graph 路径在连续周期重放时存在输出历史依赖。该结论使我们能够拒绝一个不成立的“连续 exact replay 已验证”主张，并避免基于不稳定回放作错误算法判断。它不是对实车失败根因的完整解释，也不是已经提高实车视频成功率的算法改进。

截至本报告，阶段成果可准确概括为：

| 方面 | 已完成状态 | 对实车工作的意义 |
| --- | --- | --- |
| Planner/capsule 状态契约 | 已补强并回归测试 | 避免将漏恢复状态误判为控制算法退化 |
| 单周期 exact replay（Q4-A） | 通过 | 可验证新控制器单周期重放链路 |
| 残差 CUDA 连续重放（Q4-B） | 失败并 fail-closed | 防止错误地宣称 full-method 连续精确可重放 |
| Q5/Q6 离线负向/所有权测试 | 在独立干净 root 上 47 项通过，待 Pro 终审 | 证明若干失败保护与 ring/writer 行为未回归 |
| 运动算法性能优化 | 尚未开始 | 仍是下一轮必须完成的主工作 |

---

## 2. 架构边界与本轮原则

论文主线保持不变：MPPI 仍是控制决策主体；学习模块、风险评估和 change-aware forecasting 均保留在现有架构角色内。本阶段没有以替换控制器为代价取得“优化”。

本轮明确保留且未修改的行为路径包括：

- MPPI 运动决策、候选分配与核心采样语义；
- cost 设计（含 spin/reversal 相关项）；
- mode-conditioned MPPI；
- reverse-token 激活；
- Actor/HSS 权限边界；
- residual-shield 决策策略；
- runner、HIL、发布、实车命令与实车部署。

因此，任何“实车会跑得更好”或“下一次一定能定位所有问题”的结论都不应从本阶段推导出来。

---

## 3. 已实施的工程与审计优化

### 3.1 Planner replay/capsule 状态完整性

已在 planner replay 相关链路中补强并测试以下内容：

1. **可恢复的 planner 预规划状态**：标准 MPPI 与 Paper RL-Driven MPPI 的 pre-plan 状态可显式导出/恢复，避免依赖反射或隐含进程状态。
2. **递归策略状态**：`PaperDirectControlPolicy` 的递归观测、normalizer 与 actor buffer 状态纳入恢复契约。
3. **reference 与输入绑定**：reference geometry、progress、局部障碍/预测输入、RNG 与 provenance 的完整性被检验；错误 source/artifact/config/RNG/reference 身份应 fail-closed。
4. **缓存与别名拓扑**：缓存重算、不可兼容组件、跨支路 alias topology 与动态 mutable state 被纳入测试，降低“状态看似相同但内部对象语义已漂移”的风险。
5. **post-plan 指纹**：标准 MPPI 与 Paper RL-Driven MPPI 输出的计划后指纹、sample identity、warm-start prior 和 nominal fallback 被纳入验证。

这些改动的作用是提升**诊断可信度**：当离线重放失败时，能够先排除状态遗漏、reference 漂移或缓存污染，再讨论控制算法是否存在真实问题。

### 3.2 Ring/window/writer 所有权与耐久性

对 planner replay capsule 的环形缓冲与 writer 生命周期补充了 fail-closed 测试覆盖，包含：

- generation-bound slot lease；
- 重复释放、过期 lease 和 schema 不匹配拒绝；
- 整个窗口的原子接受/拒绝；
- early trigger 记录而非抛出；
- capture gap 对当前与保留 lease 的释放；
- shutdown 时 active/history/queued lease 的释放；
- writer 失败不使 close 死锁，且写入 durable overflow/error/timeout 状态；
- 50-slot ring 的 4 MiB 上界和 writer queue 的 8 MiB 上界。

它们尚未接入运行时 runner；因此这些是离线工程保障，不是已启用的实车采集功能。

### 3.3 source/provenance 与工作区隔离

本阶段将不同证据身份明确分开：

| 身份 | SHA | 含义 |
| --- | --- | --- |
| 已资格化运行时来源 | `0b65cefe6bc58a6ef71a91877cb28555a6f3a573` | Q2/Q3/Q4 初始运行时证据绑定来源 |
| 审计证据冻结提交 | `a6efe039637588b74764abdea05c73f6c25d5327` | 仅增加审计记录和回归测试，不是新运动算法实现 |
| 证据冻结树 | `07f0dee904e71067e98902d4a3b71fc62e3df981` | 独立资格 root 使用的 Git tree |

原工作区出现了内容等价但表示不干净的状态：`models.py` 的 working-tree blob 与 `HEAD` blob 一致、暂存与未暂存 diff 为空，但 `git update-index --refresh` 失败且 porcelain 仍显示修改。该 checkout 已被保存为异常证据，且未继续用于新的资格结果。

随后创建独立 no-checkout clone，并固定：`core.autocrlf=false`、`core.eol=lf`、`core.fsmonitor=false`、`core.untrackedCache=false`。该独立 root 验证通过：

- `HEAD=a6efe039...`；
- `update-index --really-refresh=0`；
- staged/unstaged diff 均为 0；
- porcelain v2 为空；
- `models.py` 工作树 Git object 与 `HEAD` blob 均为 `4f9a4f7c8dcdcaf0a65acac43b15a0b41881986f`；
- index/working-tree EOL 均为 LF；
- 工作树原始 SHA-256 与 `git cat-file blob HEAD:models.py` 一致；
- CUDA 解释器导入的 `mobile_robot_mppi` 模块均来自该独立 root。

此项隔离只用于提高离线证据可靠性；它不改变控制算法。

---

## 4. CUDA residual 连续重放发现与防护

### 4.1 资格结果

| Gate | 结果 | 含义 |
| --- | --- | --- |
| Q4-A：fresh-controller 单周期 exact replay | **PASS** | 单周期恢复链路可精确重放 |
| Q4-B：连续周期 residual 分支 exact replay | **FAIL** | 第二周期 residual 预测轨迹出现首个分歧 |
| Q4-C：该 residual CUDA signature 的后续连续资格 | **BLOCKED** | 不得以该 signature 声称 full-method 连续 exact replay |

形式化处置为：

```text
Q4_A_ONE_CYCLE_EXACT_REPLAY_PASS
Q4_B_OFFLINE_CONSECUTIVE_EXACT_REPLAY_FAIL
Q4_C_BLOCKED_FOR_THIS_SIGNATURE
CUDA_GRAPH_REPLAY_OUTPUT_HISTORY_DEPENDENCE_CONFIRMED
RESIDUAL_CUDA_GRAPH_SIGNATURE_UNSUPPORTED_FOR_EXACT_REPLAY
```

### 4.2 已验证的机制边界

排查依次验证了 cycle-0 → cycle-1 状态边界、residual context transition 和 residual rollout input 等价性。现有证据表明，问题位于 CUDA graph replay 的输出历史依赖，而非已检查的外部 planner 输入差异。

受控 regression 使用语义键 `(1, 36, 2)`：

- zero-poison replay 精确；
- finite-pattern poison 会改变输出张量 `(1, 37, 5)` 的 185 个元素中的 180 个；
- 因此该 signature 不能获得 `EXACT_REPLAY_VERIFIED` 的连续重放标签。

### 4.3 已实施的防护

提交 `a6efe039` 中新增：

- `docs/audit/residual_cuda_graph_exact_replay_disposition.json`：机器可读处置记录；
- `tests/learning/test_residual_device_rollout.py`：finite poison 与禁止错误 exact-replay 标记的回归保护。

这项防护不修复 CUDA wrapper，也不改变车辆控制输出；它避免后续优化用错误的“可复现”前提做决策。

---

## 5. 已运行的离线测试与结果

在独立 clean root、现有 CUDA Python 环境下运行：

```text
python -m pytest \
  tests/real_robot/test_planner_replay_capsule.py \
  tests/learning/test_residual_device_rollout.py -q
```

结果：**47 passed，20 个 PyTorch deprecation warnings，6.87 s**。

该测试集覆盖 capsule 完整性、typed incomplete capsule、窗口/lease/writer 生命周期、reference/artifact 绑定、标准与 Paper RL-Driven MPPI replay contract、residual shield branch contract 和 residual device rollout 的负向 guard。

证据产物：

- `qualification_outputs/a6efe039_q5_q6/representation_anomaly_bundle.json`
- `qualification_outputs/a6efe039_q5_q6/clean_root_q5_q6_run_manifest.json`
- `qualification_outputs/a6efe039_q5_q6/q5_q6_pytest.log`
- `qualification_outputs/a6efe039_q5_q6/q5_q6_junit.xml`

Q5/Q6 的最终技术接受权仍属于 Pro；本报告只陈述已执行的 clean-root gate 与测试结果。

---

## 6. 本阶段没有完成的优化

以下工作没有实施，不能算作已优化：

1. 直接提高实车轨迹跟踪、避障、速度平滑、转向稳定性或通过率的运动算法修改；
2. 针对 spin/flip、反向、走廊、对向会车或动态障碍行为的 causal 机制修复；
3. MPPI cost/采样/候选分配重设计；
4. change-aware forecast 的在线时间对齐或观测时延补偿；
5. Actor/HSS 或 residual-shield 策略权限重划分；
6. runner、HIL 或真实车辆闭环采集接入；
7. 用一次实车实验即可精确归因所有问题的能力。

因此，本阶段的正确价值是“避免错误诊断与虚假资格结论”，而不是“实车控制性能已经优化完成”。

---

## 7. 下一阶段：失败定位与论文保持型算法优化

用户已要求终止继续沉入资格审计细节。下一步应由 Pro 基于论文、源码、根目录实验日志和既有动态障碍/走廊/反向行为材料，选择第一个有明确机制假设的 intervention。

建议交付顺序：

1. **失败机制选择**：从真实运动现象或最接近的离线失败日志中，明确一个可证伪假设；
2. **最小代码改动**：指定模块、函数、输入输出契约与保持不变的论文组件；
3. **离线验收指标**：将改动与实际可见行为关联，例如低速抖动/旋转次数、最小障碍间距、轨迹进展、控制变化率、失败模式触发率；
4. **批准后的非发布验证**：只在 Pro 明确授权后做定向仿真、离线 replay 或后续硬件验证；
5. **实车前检查表**：固定版本、配置、模型、实验场景和日志记录字段，避免一次视频无法解释。

目标不是再增加审计机制，而是在现有 Learning-Augmented Risk-Aware MPPI 主框架内获得一个可解释、可测试、且面向下一次实车视频的算法改动。

---

## 8. 交接材料

已生成供 Pro 全面审计的压缩包：

`D:/Projects/RL_MPPI_PRO_ALGORITHM_AND_FAILURE_AUDIT_20260816_v2.zip`

其中包括：论文 PDF、干净源码快照、核心实验日志、近期审计证据，以及 `PRO_ALGORITHM_AUDIT_BRIEF.md`。交接说明明确要求：**保留已完成优化，停止扩张审计工作，开始实车失败定位或论文保持型算法优化。**

---

## 9. 结论

本阶段没有完成“让车更好地运动”的算法优化，但完成了必要且已验证的回放可靠性、状态契约、负向保护和干净证据根隔离。其价值在于：后续算法调参不再建立在一个已知会被 residual CUDA graph 历史依赖污染的错误重放结论之上。

从现在开始，项目的成功标准必须转为：由 Pro 选定具体实车失败机制，并在论文架构不变的条件下实施和验证一项能够改善下一次实车表现的算法级干预。
