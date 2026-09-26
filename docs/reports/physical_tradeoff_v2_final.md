# Physical tradeoff v2 final — NO-GO at oracle headroom

## Material Passport

独立 protocol，2026-09-07。用户指定科学问题和阶段顺序；代理实现、执行、分析。
分支 `codex/physical-tradeoff-v2`，父提交 `0c6dfdab49ba18921401b9b9141684fa96a6b1e0`。
本报告不覆盖上一轮 physical_tradeoff_sac_mppi，不修改其 Phase-4 NO-GO。
实验是 MuJoCo 仿真，CPU 实测计算延迟进入真实仿真状态转移；没有实机验证。

## 决策

**NO-GO：本轮不训练 SAC。** 留出 seed 的 context selector 相对 global fixed，
零计算价格 headroom 为 **4.041%**，扣除预冻结 0.5 百分点
物理 overhead 敏感性预留后为 **3.541%**，
预冻结 strong-GO 门槛为 **10.0%**。
阈值、网格和评估种子均在生成最终网格前存档，没有事后放宽。

H 的闭环敏感性成立，K 的采样/任务收益因场景而异；这些机制证据没有自动
转化为足够大的 context-level 自适应收益。闭环固定延迟实验亦未证明有意义的
speed × delay 任务退化，不能以 staleness 作为本轮核心贡献。

- context_choices_stable: **True**
- context_choices_differ: **True**
- at_least_three_positive_folds: **True**
- bootstrap_lower_positive: **True**
- reaches_threshold: **False**

## 直接回答 15 个问题

1. **K 是否存在 closed-loop trade-off？** 有场景依赖的任务响应；Medium-high 的
   K16→256 大幅改善，测试上界未见自然劣化。Easy 的高 K 没有收益，低速 J 略增。
   尚未证明由物理延迟造成的内部 K 最优点；不能把计算增加本身叫物理性能下降。
2. **H 是否存在 closed-loop trade-off？** 是，就冻结控制器而言。K128 下 Medium-high
   H16/H20 均 4/4 成功，H8/H12/H40 均 0/4；H30 仅 1/4。短视/长 H 失败模式可重复，
   但失效原因尚未被单独识别，不能把整个效应归因于 rollout error 或延迟。
3. **物理 latency 是否改变任务表现？** 数值轨迹会变化；0/20/50/80 ms 闭环处理未达到
   预冻结有意义任务退化门槛。开放环命令重放出现终点失败，仅支持重放任务的敏感性。
4. **高速是否对同样 latency 更敏感？** 几何漂移更大，但闭环任务退化的速度交互不稳定，
   未获支持。速度标签是 command cap，Medium-low 常停滞，不能当成严格固定实际速度。
5. **补偿后剩多少？** 高 H 失败基本保留；Medium-high H40 off/on 都 0/4 成功。
   逐格 J 变化见下表。它排除不了采样/目标函数原因，也不证明延迟不可补偿。
6. **最优 K/H 随 scene 变化？** 描述性选择会变化；留出选择的稳定性见下表。
   标签选择收益未过 gate，不能宣称已得到可靠可部署的最优分配函数。
7. **随 speed 变化？** 网格最优标签会变化，但实际速度受到闭环行为影响；该关联不是
   速度单因素因果结论，且没有足够净收益支撑本轮学习。
8. **Oracle headroom？** 零价格 4.041%；保守净值
   3.541%。global J=2655.80，
   context J=2548.50。它是粗网格、episode 标签选择基准，
   不是所有在线时变策略的数学上界。
9. **达到冻结阈值？** 否。阈值 10.0%，同时要求
   留出收益、bootstrap 下界和 context 选择稳定性；完整布尔检查在上方。
10. **充分理由训练 SAC？** 没有，按用户 gate 停止。即使知道 scene/speed 标签，
    当前粗网格中的稳定可回收收益仍不足。
11. **SAC 回收多少 headroom？** 未训练，不适用；没有伪造学习曲线或策略热图。
12. **SAC-KH 优于 Bøhn SAC-H？** 未评估。保留 Bøhn 2021 SAC/tanh/环境 rounding、
    replay 连续动作和 terminal-value 匹配原则；未用后续 PPO2 冒充原 SAC。
13. **Heuristic 足够？** 未拟合在线 heuristic。甚至带已知标签的离线选择都未过门槛，
    当前优先使用经验证的固定预算；不能据此宣称所有 heuristic 或动态策略都无收益。
14. **ICODE/reliability 值得加入？** 不作为当前失败 gate 的补救。可独立研究创新量、
    disagreement 等是否预测 H 的边际收益，必须使用新 protocol；不假定 ICODE-on
    等于可靠，也不假定开启后应增加 H。旧预测 artifacts 全保留。
15. **足以形成 ICRA 核心 claim？** 不足以支持物理延迟驱动 SAC 联合 K/H 分配的核心
    性能贡献。已有证据适合机制/负结果记录；若未来研究先修复 nominal Medium-low
    停滞或采样退化，应在新协议重新验证 headroom，不能复用本轮作阳性结果。

## 协议、实现与规模

- 主协议：`docs/protocols/physical_tradeoff_v2.md`；明确的 pre-grid 开发修订：
  `docs/protocols/physical_tradeoff_v2_landscape.md`；机器可读阈值：`landscape_freeze.json`。
- Nominal dynamic-unicycle (.18/.12 s)、RK4；没有 checkpoint，没有 reliability ensemble。
  Easy/Medium 为已有空场/对角线单圆柱 benchmark，不新造阳性场景。
- K128 初始串行 576 次 profile 复核通过；随后 A 四个 development seeds。
  速度上限 .25/.65 m/s，控制周期 .1 s，最长36 s，安全前缀8不变。
- 决策 timer 包含 context、预算、planner、补偿、安全仲裁与控制器通知；
  timer 到 plant.step 前结束。仿真执行、传感器合成和离线诊断不计入决策耗时。
  物理 wrapper 在 tau 内延续旧命令，保留40 ms actuator transport，超期丢弃新命令。
- 当前 pose/twist 是明确的无噪声 ground-truth sensor isolation，不是 odometry 部署证明。
  真值用于执行 cost/诊断；不把当前完成后的 tau/ESS 输入当前决策。
- J 为真实执行 running cost 和碰撞剩余步惩罚；本轮无碰撞时后项为零。
  obstacle 项沿用 MPPI 的激光局部障碍表示和系数；它不是与感知密度无关的几何风险函数。
  没有 potential-shaped return、stale penalty，也没有用 K/H/KH 代替实际计时。
- 超时不另加一次 failure penalty；持续 goal cost 已累计。失败可与较低 J 并存，
  因此所有成功、终点距离和路径长度与 J 一起保存，尤其不得只按 J 美化停滞。
- 所有 seed 共享固定几何/起点；配对 seed 主要改变 MPPI 随机采样，
  并不是独立随机场景样本。最终 grid 新 seed 7092861–7092864，不复用开发种子。
- H-step error 是每10周期从 sensed state 出发，按未来**已执行 actuator 分段命令**
  离线推进 nominal model 到 H；消除了未来 replanning 命令不同的混淆，但各 H 访问的
  状态/运动分布不同、末尾窗口被截断，不能据此独立证明长 H 的因果误差增长。

| Stage | Episodes |
| --- | --- |
| phase_a_confirm | 72 |
| phase_a_screen | 24 |
| phase_b_closed | 64 |
| phase_b_replay | 64 |
| phase_c | 192 |
| phase_d | 64 |
| phase_e | 256 |

共核验 **736 episodes / 165482 cycles**。另有576次计时、
32个独立采样种子/可达状态/K 的 sampling probes、256次无训练网络 overhead probe。
所有运行串行，Torch/BLAS threads=1，i9-14900HX CPU；硬件/版本/源文件 hash 在各阶段 manifest。

## Phase A：H 闭环结果

下表的 SD 是四个 seed 的样本标准差，不是 cycle 作为独立样本的虚假置信区间。
Medium-low 所有 H 都失败，该 context 的代价最优仅表示失败程度不同。

| Context | H | Success | Mean J ± seed SD | Duration s | Final distance m | Mean latency ms | ESS/K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| easy/high | 8 | 4/4 | 595.0 ± 12.7 | 8.93 | 0.289 | 1.83 | 0.619 |
| easy/high | 12 | 4/4 | 603.2 ± 0.8 | 9.03 | 0.288 | 2.13 | 0.524 |
| easy/high | 16 | 4/4 | 595.3 ± 11.2 | 8.95 | 0.294 | 2.65 | 0.435 |
| easy/high | 20 | 4/4 | 609.0 ± 10.5 | 9.15 | 0.286 | 2.93 | 0.338 |
| easy/high | 30 | 4/4 | 604.2 ± 8.3 | 9.12 | 0.287 | 4.61 | 0.175 |
| easy/high | 40 | 4/4 | 638.2 ± 12.6 | 9.83 | 0.294 | 5.21 | 0.077 |
| easy/low | 8 | 4/4 | 1195.7 ± 3.4 | 18.77 | 0.296 | 1.76 | 0.725 |
| easy/low | 12 | 4/4 | 1182.5 ± 0.7 | 18.60 | 0.296 | 2.26 | 0.585 |
| easy/low | 16 | 4/4 | 1166.6 ± 6.1 | 18.48 | 0.289 | 2.74 | 0.457 |
| easy/low | 20 | 4/4 | 1145.9 ± 3.6 | 18.23 | 0.292 | 3.32 | 0.367 |
| easy/low | 30 | 4/4 | 1120.5 ± 3.1 | 17.82 | 0.290 | 4.35 | 0.257 |
| easy/low | 40 | 4/4 | 1128.7 ± 6.5 | 17.73 | 0.291 | 5.13 | 0.202 |
| medium/high | 8 | 0/4 | 11742.7 ± 234.3 | 36.00 | 3.064 | 1.90 | 0.267 |
| medium/high | 12 | 0/4 | 11257.4 ± 768.0 | 36.00 | 3.064 | 2.46 | 0.050 |
| medium/high | 16 | 4/4 | 1789.9 ± 49.3 | 10.50 | 0.284 | 2.76 | 0.207 |
| medium/high | 20 | 4/4 | 1721.5 ± 111.0 | 10.73 | 0.291 | 3.11 | 0.166 |
| medium/high | 30 | 1/4 | 20633.8 ± 11320.4 | 30.52 | 2.182 | 4.60 | 0.020 |
| medium/high | 40 | 0/4 | 26407.1 ± 258.1 | 36.00 | 2.802 | 5.32 | 0.008 |
| medium/low | 8 | 0/4 | 9118.2 ± 39.3 | 36.00 | 3.126 | 1.87 | 0.381 |
| medium/low | 12 | 0/4 | 7230.6 ± 88.3 | 36.00 | 3.203 | 2.56 | 0.131 |
| medium/low | 16 | 0/4 | 7304.5 ± 331.7 | 36.00 | 3.190 | 2.84 | 0.043 |
| medium/low | 20 | 0/4 | 8420.5 ± 373.1 | 36.00 | 3.128 | 3.09 | 0.026 |
| medium/low | 30 | 0/4 | 17231.5 ± 1062.2 | 36.00 | 2.733 | 4.30 | 0.014 |
| medium/low | 40 | 0/4 | 21093.8 ± 227.2 | 36.00 | 2.790 | 5.72 | 0.011 |

H40 Medium-high ESS/K 约.01，提示权重退化。长 H 失效有强证据，但采样数量固定、
temperature固定、running/terminal cost 比例随 H 改变，多个机制仍混合。

## Phase B：speed × injected delay

| Context | Delay ms | Paired mean ΔJ/J | Meaningful task effect |
| --- | --- | --- | --- |
| easy/low | 20 | +0.18% | False |
| easy/low | 50 | +0.41% | False |
| easy/low | 80 | +1.09% | False |
| easy/high | 20 | -0.54% | False |
| easy/high | 50 | +1.82% | False |
| easy/high | 80 | +0.36% | False |
| medium/low | 20 | +2.93% | False |
| medium/low | 50 | +3.27% | False |
| medium/low | 80 | +3.62% | False |
| medium/high | 20 | +1.02% | False |
| medium/high | 50 | +0.56% | False |
| medium/high | 80 | +1.11% | False |

每个处理都用旧命令实际推进 MuJoCo，不 sleep。闭环对照命令会因状态变化而反馈调整。
额外重放严格共享 high-speed 零延迟命令 bank / [vmax,1.25]，然后按两速度上限缩放，
360周期，不足长度补零；重放不重新做 safety/policy 决策，发生碰撞仍终止。
同一时间轴上低速缩放并不保证覆盖同一空间轨迹；该隔离诊断不能冒充公平闭环任务。
其中高速 Easy/Medium 的部分终点成功丢失对延迟敏感，低速 bank 自身未必能完成。
完整 path deviation、角度/速度 stale、success/collision 在 B 原始与 decision 文件。

## Phase C：冻结延迟外推

| Context | H | Off J | On J | Improvement | Off/on success |
| --- | --- | --- | --- | --- | --- |
| easy/low | 8 | 1196.2 | 1195.9 | +0.03% | 4/4 → 4/4 |
| easy/low | 12 | 1182.7 | 1182.0 | +0.06% | 4/4 → 4/4 |
| easy/low | 16 | 1165.3 | 1166.1 | -0.07% | 4/4 → 4/4 |
| easy/low | 20 | 1146.2 | 1146.5 | -0.03% | 4/4 → 4/4 |
| easy/low | 30 | 1119.0 | 1120.7 | -0.15% | 4/4 → 4/4 |
| easy/low | 40 | 1128.0 | 1129.4 | -0.12% | 4/4 → 4/4 |
| easy/high | 8 | 609.0 | 601.6 | +1.21% | 4/4 → 4/4 |
| easy/high | 12 | 594.6 | 594.9 | -0.06% | 4/4 → 4/4 |
| easy/high | 16 | 601.4 | 596.8 | +0.76% | 4/4 → 4/4 |
| easy/high | 20 | 600.1 | 606.9 | -1.14% | 4/4 → 4/4 |
| easy/high | 30 | 617.2 | 625.7 | -1.36% | 4/4 → 4/4 |
| easy/high | 40 | 633.0 | 630.8 | +0.34% | 4/4 → 4/4 |
| medium/low | 8 | 9102.1 | 9098.6 | +0.04% | 0/4 → 0/4 |
| medium/low | 12 | 7222.0 | 7225.6 | -0.05% | 0/4 → 0/4 |
| medium/low | 16 | 7279.6 | 7291.3 | -0.16% | 0/4 → 0/4 |
| medium/low | 20 | 8423.7 | 8419.0 | +0.06% | 0/4 → 0/4 |
| medium/low | 30 | 17115.5 | 17146.2 | -0.18% | 0/4 → 0/4 |
| medium/low | 40 | 20712.3 | 20963.1 | -1.21% | 0/4 → 0/4 |
| medium/high | 8 | 11886.7 | 12384.8 | -4.19% | 0/4 → 0/4 |
| medium/high | 12 | 10881.7 | 10935.9 | -0.50% | 0/4 → 0/4 |
| medium/high | 16 | 1756.5 | 1835.6 | -4.51% | 4/4 → 4/4 |
| medium/high | 20 | 1690.0 | 1709.8 | -1.18% | 4/4 → 4/4 |
| medium/high | 30 | 26380.1 | 26010.3 | +1.40% | 0/4 → 0/4 |
| medium/high | 40 | 26548.9 | 27014.7 | -1.75% | 0/4 → 0/4 |

所有 off/on 都是新的随机排序配对执行；tau_hat 来自 A 前 profile、仅依赖 K/H。
不能使用当前 tau。高 H 主要劣化未消失，但 B 已削弱延迟因果解释。
没有高 K 的补偿扫描：D 在 Medium 显示 K 到256仍改善，没有已识别的高 K downside
需要“被补偿消除”的证据；不能外推到更高 K 或更慢硬件。

## Phase D：K 闭环结果

| Context | K | Success | Mean J ± seed SD | Mean latency ms |
| --- | --- | --- | --- | --- |
| easy/high | 16 | 4/4 | 592.2 ± 4.1 | 2.07 |
| easy/high | 64 | 4/4 | 604.6 ± 10.9 | 2.50 |
| easy/high | 128 | 4/4 | 609.8 ± 8.8 | 2.71 |
| easy/high | 256 | 4/4 | 606.9 ± 14.5 | 3.09 |
| easy/low | 16 | 4/4 | 1117.6 ± 2.9 | 2.17 |
| easy/low | 64 | 4/4 | 1153.5 ± 4.6 | 2.36 |
| easy/low | 128 | 4/4 | 1167.7 ± 7.2 | 2.62 |
| easy/low | 256 | 4/4 | 1175.8 ± 2.2 | 3.04 |
| medium/high | 16 | 2/4 | 15352.5 ± 10229.4 | 2.19 |
| medium/high | 64 | 4/4 | 4842.2 ± 1361.9 | 2.43 |
| medium/high | 128 | 4/4 | 1797.4 ± 91.6 | 2.82 |
| medium/high | 256 | 4/4 | 1598.5 ± 105.1 | 3.37 |
| medium/low | 16 | 0/4 | 7702.4 ± 712.8 | 2.14 |
| medium/low | 64 | 0/4 | 7435.7 ± 558.7 | 2.35 |
| medium/low | 128 | 0/4 | 7274.4 ± 329.6 | 2.55 |
| medium/low | 256 | 0/4 | 6953.0 ± 109.0 | 3.24 |

独立32 seed采样的聚合 raw-first-action 方差 K16=0.012287、
K64=0.007435、K128=0.005874、K256=0.002582，
K16→256 降低 **78.99%**。状态来自固定 H16 参考轨迹的
0/1/2 m 投影进度首次到达点，不可达点明确缺失；每次 probe 重置全部 planner 状态，
采用冷 warm-start 且 previous command固定，不混入前一个采样 run 的记忆。
U_MC 使用已有样本的四组局部权重方差，组内样本数K/4；并非 full-K 方差的校准估计。

## Phase E 与 oracle

256 episodes，4K ×4H ×4contexts ×4fresh seeds。H由开发阶段冻结为8/16/20/40，
保留中间成功点与两端；H12的省略及选择理由在 pre-grid 修订中公开。
未按 final grid 改模型、cost、seed或threshold。

| Context | Modal K/H | Mode frequency | All training-fold choices |
| --- | --- | --- | --- |
| easy_low | [16, 20] | 3/4 | [[16, 40], [16, 20], [16, 20], [16, 20]] |
| easy_high | [128, 20] | 3/4 | [[128, 20], [128, 20], [256, 8], [128, 20]] |
| medium_low | [256, 16] | 4/4 | [[256, 16], [256, 16], [256, 16], [256, 16]] |
| medium_high | [256, 20] | 4/4 | [[256, 20], [256, 20], [256, 20], [256, 20]] |

| Held-out seed | Selected global K/H | J global | J context | Headroom |
| --- | --- | --- | --- | --- |
| 7092861 | [256, 16] | 2601.31 | 2562.19 | +1.504% |
| 7092862 | [256, 16] | 2594.89 | 2564.68 | +1.164% |
| 7092863 | [256, 16] | 2606.30 | 2578.21 | +1.078% |
| 7092864 | [256, 16] | 2820.71 | 2488.90 | +11.763% |

| Compute price | J global | J context with priced overhead | Headroom | Net after reserve | 95% seed bootstrap gross |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 2655.80 | 2548.50 | +4.041% | +3.541% | [+1.121%, +9.276%] |
| 0.1 | 2656.35 | 2549.07 | +4.038% | +3.538% | [+1.120%, +9.274%] |

四折零价格 gross 收益分别约1.50%、1.16%、1.08%、11.76%；整体4.04%的收益较多
来自最后一折 Medium-high 的较大差值。四折都为正不代表稳定达到10%的实际收益。
两个 selector 的16个留出context episodes均为12成功/4超时，成功率没有增加；
共同失败全部来自 Medium-low，不能把这些均值最优称为该场景已解决。

开发逐格 CV 中位数 **1.764%**，门槛
max(10%,2×CV)=10.0%。所有逐格CV也已公开，
该稳健汇总不会掩盖高方差失效 cells。bootstrap仅四个 seed cluster，精度有限。
Actor+switch P99 **0.118 ms**，context P99
**0.181 ms**；context 已计入固定预算时延，
oracle 额外 actor/switch 按价格单列。0.5百分点物理 overhead预留是敏感性假设，
不是实测因果上界；即使完全去掉，也应对照上表 gross headroom 判断。

## 验证与可复现性

审计结果 **True**：逐 episode 重算 cost、路径、时长、距离和统计，
验证计时分段求和、物理.1s周期、deadline hold、冻结补偿 lookup及命令归一化一致性。
四个不同context 的H16原始控制/实际延迟做确定性物理重放，最大状态差见 audit_verified/decision.json。
首次审计在空场 clearance=Infinity 被 JSON 转成 null 的比较处停止，修正审计器后独立重跑；
失败尝试保存在 audit/，没有更改任何实验数据。
这不是声称重新获得相同操作系统 wall-clock latency。
测试执行结果单独归档 validation/；图形脚本、所有 cycle JSONL.gz、model error、
配置、种子/顺序、commit/source hashes与原始用户请求均在独立 v2 artifact root。
全套回归测试 **901 passed，51.46 s**；JUnit XML保存在 validation/pytest.xml。
与父提交比较，旧v1代码、protocol、报告、artifact目录均无差异。

## 图形交付

所有完成阶段提供 PNG 300dpi 与矢量 PDF，目录 `research_artifacts/physical_tradeoff_v2_2026-09-07/figures/`。

| Requested figure | Artifact/status |
| --- | --- |
| 1 H closed-loop | h_task, h_duration, h_final_distance |
| 2 H model error | h_model_error |
| 3 H latency/staleness | h_latency, h_staleness |
| 4 K sampling variance | k_sampling_variance |
| 5 K closed-loop | k_task |
| 6 K latency | k_latency |
| 7 Speed × delay | phase_b_closed_stale/task; phase_b_replay_stale/task |
| 8 Compensation | compensation |
| 9 Joint landscape | joint_landscape |
| 10 Context optima | context_optima |
| 11 Oracle | oracle_headroom |
| 12 SAC policy map | Not run: oracle gate failed |
| 13 SAC-KH vs SAC-H | Not run: oracle gate failed |
| 14 Performance–latency | joint_performance_latency; h_performance_latency |

SAC T0/T1/T2/T3、SAC-H、terminal-value fidelity variant、拟合在线 heuristic均未运行。
本轮在 oracle gate 正式停止，没有为了全流程继续训练。建议保留固定预算证据和
H/ESS退化诊断；只有另立明确问题并修复任务/采样局限后，才重新讨论学习分配。
