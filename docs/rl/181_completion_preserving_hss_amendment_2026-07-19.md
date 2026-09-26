# Full Proposed 开发门失败与单次补救修订（预注册）

日期：2026-07-19  
状态：在解封确认集种子 51–55 之前冻结  
关联方案：`docs/rl/180_full_proposed_factorial_prereg_2026-07-19.md`

## 1. 修订原因

Full Proposed 的 3-seed 开发实验（种子 48–50）在
`clean_single_obstacle` 的三个物理域上完成了 36 个随机区组 episode。
固定 30% Actor Hybrid Sampling 的两个方法均为 9/9 成功，而可靠性自适应
HSS 出现以下三个非碰撞失败：

| seed | 方法 | 物理域 | 最近目标距离 | 结束距离 | 失败时低可靠性占比 |
|---:|---|---|---:|---:|---:|
| 48 | value-aligned + adaptive HSS | combined unseen | 0.305 m | 1.757 m | 0.720 |
| 49 | ordinary ICODE + adaptive HSS | nominal seen | 0.337 m | 1.444 m | 0.950 |
| 49 | ordinary ICODE + adaptive HSS | long-delay seen | 0.466 m | 1.607 m | 0.973 |

三个失败 episode 在接近目标阶段的实际 Actor 引导比例均为 0。车辆已经接近
0.30 m 成功阈值，但随后越过目标并继续远离。该证据否定了“低可靠性时在所有
任务阶段都允许 Actor 引导完全退化到 0”的实现，而没有否定：

1. ICODE 残差动力学；
2. value-aligned ICODE；
3. 可靠性估计；
4. RL-Driven MPPI 的固定 Actor/critic 接口；
5. 论文冻结的 ICODE × RL 双向耦合方向。

开发集的用途正是发现并修正这种机制缺口。确认集种子 51–55 尚未读取或运行。

## 2. 唯一允许的补救改动

保持原来的可靠性映射：

\[
\rho_{\mathrm{raw}}(c_t)\in\{0,\ 0.30,\ 0.60\},
\]

仅在机器人进入当前目标的终段邻域后加入完成性下限：

\[
\rho_t =
\begin{cases}
\max\left(\rho_{\mathrm{raw}}(c_t),\rho_{\min}\right),
& \lVert p_t-p_g\rVert_2 \le R_{\mathrm{terminal}},\\
\rho_{\mathrm{raw}}(c_t), & \text{otherwise}.
\end{cases}
\]

冻结参数：

```text
rho_min = 0.30
R_terminal = v_max * H * dt
           = 0.35 m/s * 36 * 0.10 s
           = 1.26 m
```

`R_terminal` 不是从三个失败轨迹拟合得到，而是 MPPI 在无减速假设下一个预测
时域的最大平移可达距离。`rho_min` 等于已在相同预算下获得 9/9 成功的论文式
固定 Actor 引导比例。终段下限只改变 100 条候选中的来源分配，不增加 rollout
总数、不改变 ICODE、critic、MPPI 代价、LaserScan、scan_guard 或安全仲裁。

## 3. 因果与安全约束

- 终段判断只使用当前观测状态与当前目标，不访问未来真实状态。
- 可靠性仍由前一控制周期估计并在下一周期使用。
- 低可靠性在终段之外仍可将 Actor 引导降为 0。
- 高可靠性仍可将 Actor 引导升为 60%。
- MPPI 始终输出最终控制；Actor 只生成候选。
- 固定 HSS 两个对照组不启用终段下限。
- terminal value 继续使用冻结的 SAC target critic；候选级 terminal-value gate
  继续关闭。
- Memory、RL online learning 和实车部署继续不进入本实验。

## 4. 冻结验证顺序

### 4.1 单元与回归测试

必须证明：

1. 终段外的低可靠性仍产生 0% Actor 候选；
2. 终段内的低可靠性产生 30% Actor 候选；
3. 高可靠性仍产生 60% Actor 候选；
4. 关闭该配置时数值行为保持不变；
5. 总 rollout 预算始终为 100。

### 4.2 开发集复验

仅重新运行原开发集种子 48–50。通过条件：

1. Full Proposed 成功数不低于 ordinary fixed；
2. Full Proposed 碰撞数不高于 ordinary fixed；
3. Full Proposed 的 seed-cluster 配对最终距离不劣于 ordinary fixed，或者成功率
   提升足以使最终距离成为次级指标；
4. 至少保留一个可审计的 ICODE/RL 机制收益（smoothness、compute 或 distance）；
5. 不再修改冻结参数。

若不通过，停止解封确认集，并将自适应 HSS 标记为当前实现下被证伪；不得继续
在种子 48–50 上搜索半径或引导比例。

### 4.3 独立确认

开发集通过后，才运行原预注册的种子 51–55。确认集的主要比较、统计单位、
bootstrap 方法、成功/安全优先级和报告规则均沿用文档 180，不因本修订改变。

## 5. 允许与禁止的结论

若开发和确认均通过，可以声称：

> Reliability-adaptive hybrid sampling requires a task-completion boundary
> condition; a horizon-derived terminal floor preserves completion while
> retaining reliability-dependent Actor authority outside the terminal region.

不得声称：

- 该下限具有稳定性或收敛性定理；
- 三个开发种子已构成正式论文证据；
- 终段下限适用于所有机器人、目标类型或路径跟踪任务；
- 只因补救后均值更好就存在显著协同效应。

