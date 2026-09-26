# Bøhn 2021 原论文算法与场景复现

> 2026-09-25 当前阶段：门控时域六作业训练及全部222条件/3,996回合/292,389步审计通过，六策略已冻结，完整84条件固定H网格验证正在运行。完整训练清单、逐回合数据及中文报告位于本轮`training_delivery/`。**尚无完整验证效果判定，测试保持封存，复现目标未完成。** 方法偏离、初始化修订、冻结标准、预算和命令见[本轮报告](../../docs/reports/bohn2021_gated_horizon_search_2026-09-25.md)。

> 前轮保守策略迭代12份模型、72条件/1,728回合验证、36条件/864回合串行计时及审计已完成。两任务两轮均未通过全种子准入；12个自适应条件成本均高于所选固定H，计时也没有稳定全种子收益。车辆25、倒立摆30的三种子固定基线无需重复补训。求解恢复及首块干预诊断均完成，不能用个别种子修复冒充核心效果。见[完整负结果](../../docs/reports/bohn2021_conservative_iteration_2026-09-24.md)、[失败机制诊断](../../docs/reports/bohn2021_solver_recovery_diagnosis_2026-09-25.md)、[完整目标核对](../../docs/reports/bohn2021_goal_requirements_audit_2026-09-24.md)。

> 2026-09-24 接续：六组分支回报校准、全部 240 个验证回合、48 条件串行计时（480 回合重复）及完整失败诊断均已完成。**数据与计时审计通过，效果门槛失败，独立测试未开启。** 两任务三个种子的校准成本均高于 min-Q actor 和独立训练固定 H，车辆每步耗时也全部更高；不能宣称复现成功。完整模型、原始轨迹、预算、命令、环境、CSV 和图表见 [本轮中文报告](../../research_artifacts/bohn2021_reproduction_2026-09-17/results/branch_calibration_2026-09-24/delivery/report_CN.md)。[失败评估及剩余路线](../../docs/reports/bohn2021_failure_assessment_2026-09-24.md)、[证据地图](../../docs/reports/bohn2021_evidence_map_2026-09-24.md) 保留负结果和数值审计中断。历史文件清点见 [inventory.md](../../research_artifacts/bohn2021_reproduction_2026-09-17/results/campaign_inventory_2026-09-24/inventory.md)。

> 2026-09-18：[最新接手记录、后续结果和工作顺序](../../docs/reports/bohn2021_handoff_2026-09-18.md)。优化轮及低熵、预览、初始分布、终端初始化实验已推进并完成；下文26组及batch64等说明是首轮历史版本，不能视为最新配置。最新结论仍未复现RL优于固定H；新增固定H干预支持时域策略是剩余损失的直接来源。

[实验结果与图表](../../research_artifacts/bohn2021_reproduction_2026-09-17/report/progress.md) · [原文/源码定位](SOURCE_MAP.md) · [固定源码快照清单](../../research_artifacts/bohn2021_reproduction_2026-09-17/sources/archives/manifest.json)

**已完成26/26组：390,000条训练经验、387,426次梯度更新、520个最终测试回合及300个中间检查点测试回合。** 全部结果通过独立核验。启用终端值时，倒立摆RL三种子平均成本379.186，最佳固定H=30为328.964；车辆RL为5230.229，最佳固定H=15为17.307。本次没有复现RL优于固定H的结论；车辆训练后期明显退化。原配置和测试集缺失，不能据此否定原论文。

## Material Passport

- 用户要求：全面检索 *Reinforcement Learning of the Prediction Horizon in Model Predictive Control* 的相关代码，尽可能复现算法和实验场景。
- 原始证据：论文 DOI `10.1016/j.ifacol.2021.08.563`；作者的 GitHub 仓库、完整历史及公开分支。
- 本地新增：缺失实验配置的重建、运行入口、可重复测试集、核验测试、报告和独立运行时适配层。
- 复现级别：**作者同期核心代码恢复＋实验配置重建**。不能称逐数字精确复现；不预设能得到论文的4%/8%改善。
- 数据：纯仿真。测试集不进入训练经验池，不据测试结果调整配置。

## 检索的重要更正

此前仅检查当前主分支时，未找到原始SAC实现。本次检查历史后，找到以下同期作者代码：

| 来源 | 固定提交 | 恢复内容 |
|---|---|---|
| [gym-letMPC / horizon](https://github.com/eivindeb/gym-letMPC/tree/3f0572e4761f6797327e48d77e2abc343ab0239c) | 2021-02-08, `3f0572e` | AHMPC、TTAHMPC、随机避障场景、随距离/预测步数增长的不确定性、终端多项式接口 |
| [stable-baselines 历史](https://github.com/eivindeb/stable-baselines/tree/1539282c8a11417b96e13040e5c4649be251dca2) | 2021-02-08, `1539282` | 作者SAC、AHMPCPolicy、独立MPC价值经验池、32步自举、二次多项式估计 |
| [do-mpc 历史](https://github.com/eivindeb/do-mpc/tree/3f4fea1d262082083ac280b5206a7ca915a6676d) | 2021-02-03, `3f4fea1` | 折扣阶段成本、按实际H折扣终端值、学习价值函数嵌入NLP |
| [rlmpcopt](https://github.com/eivindeb/rlmpcopt) | 当前公开版本 | 后续PPO元参数论文的训练/评估、模型及数据，仅作为旁证，未冒充原SAC实验 |

检索覆盖作者18个公开仓库清单、相关仓库获取到的分支及同期提交、5个Gym fork和2个rlmpcopt fork的文件树、issues、releases、Zenodo题名标识检索。全历史路径扫描最初在WSL因网络补取超时中断；随后通过Windows Git成功完成stable-baselines和do-mpc获取到的全部引用扫描（退出码0），Gym和rlmpcopt也已完成。以`history_audit_windows.json`及`history_audit.json`中的成功记录为准，不以早期截断文件为依据。来源响应保存在 `research_artifacts/bohn2021_reproduction_2026-09-17/search/`。

核心源码引用的外部 `lmpc-horizon` 目录包含原配置和测试集，但未在上述公开来源找到。GitHub全站代码搜索返回401，Software Heritage返回验证页面，通用网页搜索未给出可用证据。这些是检索局限，不是证明代码不存在。没有联系作者或对外发送消息。

补充检查了GitHub README中的完整题名和arXiv编号，以及作者公开gist列表（0项）；没有找到新的原实验代码。grep.app的配置名检索返回429，未将其当作有效的“无结果”。三份核心源码另存为保留原许可证的ZIP快照，位于`sources/archives/`，附提交号和SHA256。

## 已恢复的算法

- 使用作者TensorFlow 1版本SAC，并非替换为PPO或本项目的MPPI策略。
- 每周期输出连续动作，缩放至1–50后取整；经验池保留原始连续动作。
- Actor为32×32；critic为256×256（沿SAC常规规模，原论文未单独列出critic规模）。
- RL/MPC折扣均为0.97；学习率3e-4、batch64、replay50000、预热100步、每步一次更新来自所固定作者代码默认值，不能称论文逐项明确的超参数。
- 使用原始观测，不做归一化；设置作者实现支持的`time_aware=True`，在时间上限截断时允许自举。这两项是重建选择，缺失配置使其无法与原实验逐项核对。
- 学习独立MPC终端值，32步阶段成本目标，不包含时域成本或RL失败惩罚。
- 作者`poly`实现是 **线性项＋逐维平方项＋偏置，无交叉项**；代码未显式约束平方系数非负。因此，不能因为论文称二次模型就擅自加入PSD约束或声称代码保证凸性。
- 作者SAC的`reward_scale`实际执行 **reward / scale**，两任务分别除以0.6、0.3。此前本地MPPI改编的乘法缩放并不相同。
- 作者AHMPC在固定最大50步NLP中用`hend`屏蔽后续动力学和成本，终端折扣按所选H计算。**这不等同于构造H步NLP，实测求解时间未必随H线性变化。** 论文主指标使用H线性成本代理。

## 场景与未知配置

| 项目 | 倒立摆 | 车辆避障 | 来源 |
|---|---|---|---|
| 状态/输入 | 位置、速度、摆角、角速度；推力 | x、y、航向；线速度、角速度 | 原文公式10/11 |
| 周期/步数 | 0.04秒 / 100 | 0.1秒 / 150 | 原文 |
| 参数 | m=.2, M=.8, l=.25, g=9.81 | 0≤v≤5, -4≤ω≤4 | 原文，g常规设置 |
| 约束 | 位置±1.5m、摆角±π/2、推力±5 | 障碍半径硬约束；1.5倍半径软约束 | 原文＋作者代码 |
| 计算/失败系数 | .003 / 10 | .001 / 2 | 原文 |
| 初始状态分布 | 位置±.5，速度±1，角度±.78，角速度±1 | x=y=θ=0 | 倒立摆沿作者其他配置范围，车辆零初始；**不是找回的原实验配置** |
| 参考 | [-1,1]分段常值，每25步变化，预测可见 | 作者生成器：方向±45°，60–99参考点，约3m/s | 倒立摆变化规则补设；车辆来自horizon分支 |
| 障碍 | 无 | 半径[.5,1]、随机偏置在参考附近，**数量3为补设** | 数量未找到，分布与位置生成器来自作者 |
| 终点阈值 | 无 | 0.5m | 作者代码 |
| 感知不确定性 | 无 | 障碍x/y/r预测偏移随距离与预测步数增长 | 作者TTAHMPC实现 |

原文没有摩擦、负重或MuJoCo轮地接触。这里使用作者连续动力学仿真和非线性MPC，未加入移动机器人项目的安全层。

原始环境`reset()`会执行一次H=50的控制动作后返回首次观测。本次保留并披露这一行为；训练步数不包括此重置动作。因此15,000条训练经验不等于全部物理积分次数。

## 独立适配层的修正

原始Git工作树不修改，修改只位于`runtime.py`：

1. `reset()`读取TVP观测时步数尚为None，提前将读取索引初始化为0。
2. 原奖励通过文本替换变量，负数平方可能失去括号而改变符号。重建配置用数值变量求值，已与独立手算的机械能和跟踪成本核对。
3. 车辆`get_obj_distance()`返回中心距离，旧判断`<=0`几乎不会检出半径内碰撞。用`distance<=radius`判定，并在最终步也检查。
4. 原32步经验采样漏掉终止奖励、可能给出0步计数且未更新末端终止掩码；独立适配改为包括终止步、不跨回合，按实际步数折扣。人为回报序列测试已验证。

这是**按论文语义修正的复现**，而不是逐位重放作者潜在bug。未经修正代码的失败日志保留，作者原测试结果是否受同样问题影响无法确认。

完成后独立轨迹审计还识别到原倒立摆环境的末步边界：第100步先被标为时间上限，未再检查状态约束。此次H=10、启用终端值、测试场景5恰在第100步越界，环境标签为`steps`。结果保留原运行记录，并单独报告真实状态越界；未在实验进行中改变该逻辑。该步剩余时长为0，约束成本仍为0，但训练的截断自举语义可能受这一边界影响，未作影响程度断言。

## 实验协议

- 两任务分别训练RL种子0/1/2，各15,000条经验。
- 每任务固定H=5、10、…、50，各自用15,000条经验学习独立终端值；固定基线使用同一训练器，actor/critic更新被保留但其动作被固定H覆盖，不参与控制决策。
- 合计26组、390,000条训练经验。固定基线不是共用RL的终端权重。
- 每任务10个新生成的冻结测试回合，种子20210917，全部方法配对使用；保留JSON与SHA256。
- 每个训练结束后均评估启用与清零终端价值的两种状态。此消融不重新训练，明确是移除终端价值的干预。
- 对6组RL已保存的2500/5000/7500/10000/12500步检查点追加冻结评估，末点采用最终15000步模型，用于重建论文学习曲线；2个独立评估进程，没有额外训练或测试集选模。中途检查点在当前步梯度更新之前保存，故对应更新次数为名义步数减100；最终模型完成14901次更新。
- 测试采用确定性策略，不更新权重或经验池；脚本核对权重散列。
- 保留成本分解、约束违反、到达情况、H分布、求解器失败、轨迹和输入。
- 主队列4个进程；运行中确认资源充足后，分批提前执行H=45/50、H=35/40、H=25/30、H=20的既定任务，已有部分组结束后再补充并行任务，另有2个检查点评估进程。所有数值库限制单线程，任务文件锁及完成标记避免重复训练。因论文成本为H代理且没有计算延迟反馈，可用于控制表现比较；**所记录墙钟时延不能作为公平实时计算基准**。若后续比较实际时延，应冻结模型后串行重测。
- 每组3小时硬超时；失败被记录且不会被标为完成。已有完成组不会重复执行。

## 运行

在WSL的项目根目录运行（独立Python 3.7环境）：

```bash
PY=/home/mapples/.local/share/bohn2021-python37/bin/python
$PY experiments/bohn2021_reproduction/configure.py
$PY experiments/bohn2021_reproduction/validate.py
$PY experiments/bohn2021_reproduction/suite.py --workers 4
```

单组：

```bash
$PY experiments/bohn2021_reproduction/run.py --task vehicle --steps 15000 --seed 0 --out /absolute/path/to/new-output
```

输出位于 `research_artifacts/bohn2021_reproduction_2026-09-17/`。`results/full/status.json`是实时状态，`suite_completed.json`只有全部作业退出后才写入；`sources/provenance.json`记录固定代码版本，`sources/runtime-freeze.txt`记录运行依赖。勿把工程短测试当作15,000步训练结果。

`SOURCE_MAP.md`提供原文/作者源码的逐模块定位。运行`report.py`可刷新结果图表；`animate.py`从已记录的RL种子0、测试场景0生成回放GIF。`verify_results.py --require-complete`会检查26组的步数、实际更新次数、配置散列、20条测试轨迹/组，以及逐步成本分解。`check_replay.py --run <已完成目录>`用于重载模型、重复运行同一冻结场景并检查一致性。

额外核验：`audit_terminal.py`对两个已训练模型各取20组输入，交叉比较TensorFlow、CasADi与独立NumPy多项式值；`audit_constraints.py`直接检查测试轨迹中的物理状态和输入，区分真实越界与环境的终止标签。结果位于`results/terminal_value_audit.json`与`results/physical_constraint_audit.json`。

## 故障定位与进一步复现

后续诊断、时间对齐修正和观测尺度重建见 [DIAGNOSIS.md](DIAGNOSIS.md)。旧26组结果不变，新训练写入 `results/diagnosis` 与 `results/refined`。`run.py --aligned --scaled-obs` 启用新的核心方法复现配置；保留作者SAC和32步联合二次终端价值学习。`--test-bank`允许明确指定独立验证集，正式保留测试由 `finish_refined.py` 在模型固定后执行。

`diagnosis_report.py`汇总全部成功及失败结果，`verify_refined.py`独立重算保留测试的物理成本和约束；`check_replay.py`可用 `--bank`、`--evaluation-dir holdout_value` 和 `--audit-out`核验新的冻结回放。
