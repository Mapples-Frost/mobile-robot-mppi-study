# Full Proposed 路径跟踪迁移实验预注册

日期：2026-07-19  
状态：任何路径跟踪结果生成前冻结  
上游实现：`f2d66fa`  
关联结果：`docs/rl/182_full_proposed_factorial_confirmation_results_2026-07-19.md`

## 1. 研究问题

点目标独立确认已经支持：

1. competence-gated value-aligned ICODE 的闭环主效果；
2. reliability-adaptive persistent Actor sampling 的闭环主效果；
3. Full Proposed 相对强简单组合的成功率和终点距离优势；
4. 不支持超加性 ICODE × RL synergy。

本实验检验一个新的、尚未由上述结果回答的问题：

> 在不重新训练 Actor、critic、ICODE，不改变 MPPI 代价和采样预算时，冻结的
> Full Proposed 是否能从 point-goal 迁移到连续曲线路径跟踪？

这不是论文中五状态 bicycle path-tracking 结果的完整复现。当前真实对象仍是
MuJoCo actuated differential-drive plant，预测状态是 dynamic unicycle 5，
控制为 \((v_{\mathrm{cmd}},\omega_{\mathrm{cmd}})\)。

## 2. 数据前兼容修正

`PolylineReference` 的当前控制目标是距离机器人约 0.40--0.45 m 的前视点。
如果 completion floor 只检查机器人到“当前目标”的距离，那么 1.26 m 终段半径
会从路径起点就激活，使 adaptive HSS 退化为最少固定 30%。

在运行任何本实验数据前，冻结以下语义：

\[
\rho_t =
\max\left(\rho_{\mathrm{raw}}(c_t),0.30\right)
\]

仅当以下两个条件同时成立：

1. `target.phase in {"terminal_approach", "terminal"}`；
2. 当前机器人到 reference target 的距离不超过 1.26 m。

PointGoal 的 phase 始终是 `terminal`，因此上一轮确认的点目标行为不变。
Polyline 在剩余路径小于其既有 `terminal_approach_distance=0.90 m` 后才允许
激活下限。tracking 阶段继续允许 0/30/60% 的完整可靠性调节。

该修改：

- 不读取未来 plant state；
- 不读取全局障碍真值；
- 不改变 rollout 数量；
- 不改变 ICODE/Actor/critic 参数；
- 不改变 LaserScan、scan_guard 或 safety arbitration；
- 只修正 reference 语义，不能根据路径实验结果再调整。

## 3. 随机区组设计

四个实验臂与确认实验完全相同：

| Cell | Residual dynamics | Actor-guided sampling |
|---|---|---|
| `ordinary_fixed` | ordinary ICODE ensemble | fixed 30% |
| `value_fixed` | value-aligned ICODE ensemble | fixed 30% |
| `ordinary_adaptive` | ordinary ICODE ensemble | adaptive 0/30/60% + terminal floor |
| `full_proposed` | value-aligned ICODE ensemble | adaptive 0/30/60% + terminal floor |

路径：

1. `path_gentle_s_l43`；
2. `path_double_turn_l43`；
3. `path_slalom_l43`。

物理域：

1. `nominal_seen`；
2. `long_delay_seen`；
3. `combined_unseen`。

每个 seed × path × physics block 包含全部四个实验臂，并在 block 内使用固定
随机种子打乱运行顺序。每拍所有方法都使用：

```text
K = 100 total rollouts
2 MPPI refinement iterations
horizon = 36
dt = 0.10 s
```

路径原配置中的 360/420/480 step 上限保持不变，不根据结果延长。

## 4. 独立单位与种子

- 开发完整性种子：401–402；
- 封存确认种子：403–407；
- seed 是 bootstrap cluster；
- path 和 physics 是同一 seed 下的重复区组；
- timestep 不是独立样本。

开发集只检查实现完整性、机制未退化和明显安全错误。不得用开发结果修改网络、
阈值、权重、半径、路径或确认 Gate。开发通过后才允许运行 403–407。

## 5. 主要和次要终点

### 主要终点

1. cross-track RMSE；
2. path completion ratio；
3. success。

### 安全与工程终点

1. collision；
2. cross-track maximum；
3. tangent-heading RMSE；
4. control jerk；
5. mean/p95 planner time；
6. deadline miss rate；
7. low/medium/high reliability occupancy；
8. completion-floor active fraction。

主要比较是：

```text
full_proposed vs ordinary_fixed
```

同时完整报告：

```text
value_fixed vs ordinary_fixed
ordinary_adaptive vs ordinary_fixed
full_proposed vs value_fixed
full_proposed vs ordinary_adaptive
ICODE × HSS interaction
```

所有 effect 统一转换为正值更有利。置信区间使用 seed-cluster bootstrap
10,000 次。二元 success/collision 也先在 seed 内跨 path/domain 求平均，再按
seed 重采样。

## 6. 开发完整性 Gate

只有全部满足才解封确认种子：

1. 72 个 episode 全部完成；
2. 四臂每个 block 各出现一次；
3. 每拍 rollout budget 保持 100；
4. adaptive 两臂在 tracking phase 至少观察到一次 raw 0% authority；
5. completion floor 在 tracking phase 的 active fraction 严格为 0；
6. completion floor 至少在一个 terminal phase 被激活；
7. 没有新增碰撞；
8. 没有 NaN/Inf、缺失 trajectory 或不完整 block；
9. 不修改冻结方法参数。

该 Gate 不要求开发集指标显著改善，避免用两颗种子筛选有利结果。

## 7. 独立确认 Gate

Full Proposed 的路径迁移确认通过需要：

1. collision 不高于 ordinary fixed；
2. cross-track RMSE favorable effect 的 95% CI 下界大于 0，或者 success/
   completion 有严格有利区间且 cross-track RMSE 满足 5 mm 非劣界；
3. completion ratio 不劣于 −0.02；
4. success rate 不出现超过 1/45（一个配对 episode）的净损失；
5. control jerk 不恶化超过 5%；
6. 三条路径、三个物理域和所有失败均完整报告；
7. interaction 无论有利、空值或不利均报告。

Full Proposed 可以作为“完整包优于简单组合”通过，而不要求超加性交互。若
interaction 不利，继续禁止 synergy 表述。

## 8. 停止规则与结论边界

开发完整性 Gate 失败时，只允许修复可证明的代码/记录错误；不得基于性能选择
新阈值。确认 Gate 失败后冻结负结果，不再使用 403–407 调参。

通过只支持：

> Frozen Full Proposed transfers to the tested obstacle-free differential-drive
> path-tracking tasks under bounded MuJoCo physics shifts.

不支持：

- 论文 bicycle model 的完整复现；
- 动态障碍导航；
- 实车 path tracking；
- universal OOD generalization；
- stability/convergence guarantee；
- super-additive ICODE × RL synergy。
