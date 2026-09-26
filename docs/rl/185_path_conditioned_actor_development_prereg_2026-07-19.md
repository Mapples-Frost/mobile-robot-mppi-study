# L185 路径条件化 Actor 开发预注册

日期：2026-07-19  
状态：在任何 L185 训练或评价结果生成前冻结  
上游封存结果：`docs/rl/184_full_proposed_path_tracking_confirmation_results_2026-07-19.md`

## 1. 动机与边界

L183–L184 使用冻结的 point-goal Actor 做零样本路径迁移。Full Proposed
相对 ordinary fixed 的 cross-track RMSE 有显著改善，但四个实验臂均为
`0/45` success，且 Actor 从未在路径任务上训练。该结果不用于反向调参。

L185 只回答：

> 给 Actor 增加可观测、无障碍真值泄漏的局部路径几何，并以沿路径进度而非
> 最终目标距离训练，能否得到可迁移到未见路径的 offline SAC policy？

它不修改 ICODE 网络、value-aligned loss、MPPI 代价、采样预算、LaserScan、
local obstacle layer、scan_guard 或 safety arbitration。

## 2. 冻结方法

### 2.1 观测增量

旧观测完整保留，并由 `include_path_context` 开关控制以下六个新增量：

1. 有符号 cross-track error；
2. 路径切线 heading error 的 sine；
3. 路径切线 heading error 的 cosine；
4. 局部离散曲率；
5. 剩余弧长；
6. polyline-valid flag。

所有量只来自任务参考路径、里程计/仿真观测和已有 LaserScan。Actor 不读取
MuJoCo 障碍真值。假想 Actor/MPPI rollout 使用只读路径投影，不推进在线
reference progress。

### 2.2 奖励增量

最终点欧氏距离 progress 权重固定为 0。新增：

\[
r_t^{\mathrm{path}}
=12\,\Delta s_t
\exp\!\left[-(e_{\perp,t}/0.75)^2\right]
-2 e_{\perp,t}^{2}
-0.2 e_{\psi,t}^{2},
\]

实现中第一项写为 `12 * Δs * corridor_gate`；上式分行仅说明组成。progress
必须由单调 polyline projection 计算。离路径越远，沿路径 progress credit
越接近 0，避免通过远距离投影跳跃获得奖励。其余安全、控制变化率、碰撞和
终止项按配置原样记录。

### 2.3 训练算法

- offline policy：custom SAC，quantile critics；
- policy action：直接 \((v_{\mathrm{cmd}},\omega_{\mathrm{cmd}})\)；
- 训练步数：每个训练种子 60,000；
- checkpoint：10k、20k、30k、40k、50k、60k 与 validation-best；
- memory：关闭；
- 训练时不执行 MPPI；在线论文实验中 Actor 仍只作为 MPPI prior/value
  组件，不越过安全仲裁直接获得最终控制权限。

## 3. 数据与路径分组

训练路径：

1. acceleration straight；
2. acceleration turn；
3. sweep；
4. chicane。

开发验证路径：

1. reverse-S；
2. hairpin。

物理域：

- train：四个 `seen` 域；
- validation：四个 `seen` 域与 `combined_unseen`。

训练种子固定为 `20261901, 20261902, 20261903`。开发闭环评价种子固定为
`551–555`。尚未打开的独立确认种子固定为 `561–565`。

403–407 是 L183 的封存确认种子，永久禁止用于 L185 选择。

## 4. 开发比较与选择

先做 Actor-only 环境评价，再做 MPPI 集成评价。

Actor-only 对照：

1. 零动作/随机初始化完整性参考；
2. 原 point-goal Actor（L175）；
3. L185 10k–60k checkpoints。

MPPI 对照：

1. simple combination：ordinary ICODE + 原 point-goal Actor；
2. path-conditioned simple combination：ordinary ICODE + L185 Actor；
3. Full Proposed：value-aligned ICODE + reliability-adaptive L185 Actor。

Checkpoint 选择顺序冻结为：

1. validation collision 最少；
2. success 最高；
3. mean cross-track RMSE 最低；
4. path completion 最高；
5. mean return 最高；
6. 若仍相同，选择更早 checkpoint。

不得仅按单条路径或单一物理域挑选 checkpoint。

## 5. 开发 Gate

只有同时满足下列条件才打开 561–565：

1. 三个训练种子均无 NaN/Inf、维度漂移或不完整 checkpoint；
2. 至少一个 L185 checkpoint 在 reverse-S 与 hairpin 两条未见路径上均优于
   L175 的 mean cross-track RMSE；
3. L185 相对 L175 在 551–555 中至少 4/5 seed 的聚合 cross-track RMSE
   方向有利；
4. completion 不下降超过 0.02；
5. collision 不增加；
6. MPPI 集成时 path-conditioned Actor 相对原 Actor 不增加失败数；
7. 代码和结果完整记录 config、git SHA、seed、checkpoint、trajectory 和
   summary。

如果 60k 后 Gate 失败，冻结负结果，不在 551–555 上继续改奖励。下一条预先
允许的路线是：单独预注册 MPPI-teacher behavior initialization，再进行 SAC
微调；它必须使用新的开发种子。

## 6. 独立确认与结论边界

通过开发 Gate 后，才使用 561–565，在未用于训练/开发的新路径几何上做
paired confirmation。主终点是 cross-track RMSE、path completion 与 success；
安全终点是 collision、minimum clearance、control jerk 和 planner time。
seed 是 bootstrap cluster，timestep 不是独立样本。

即使确认通过，也只支持：

> Path-conditioned offline SAC improves the tested RL-guided MPPI stack on
> bounded differential-drive path-tracking tasks.

不支持 bicycle 全复现、动态障碍泛化、实车结论、普适 OOD 保证、稳定性/
收敛性定理或超加性 ICODE×RL synergy。
