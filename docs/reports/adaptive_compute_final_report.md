# Model-Reliability-Aware Compute Allocation for MPPI：阶段性最终研究报告

日期：2026-09-07。结论：**NO-GO，当前结果不足以支撑 ICRA 核心 claim 或据此重构论文。**

后续最后一次挽救诊断已完成：216条配对预算分支的跨采样收益仅提高2.45%，低于预先冻结的10%标准。
按用户约定，现已停止该方向，详情见[最后一次 headroom 诊断与停止决定](adaptive_compute_headroom_final.md)。
该补充保留了17.47%的局部分支规划时间节省信号，没有将未达标等同于零收益；以下历史结果不变。

这里的 NO-GO 指当前方法、训练方案、任务和平台上的证据不足，**不是证明一般的
model-reliability-aware value-of-computation 思想永远无效**。本轮已经执行代码验证、
来源核验、baseline 移植、可靠性／场景 gate 及修订、204 个 fixed landscape cells、
单 seed PPO 开发训练和配对 baseline 诊断。所有 T1 版本未通过预先规定的升级条件，
因此没有启动 T2 或正式 8–12 seed T3，也没有伪造正式置信区间／显著性结果。

## Material Passport

- 用户提供研究问题、方法边界、baseline 列表、实验纪律及分级流程。
- AI 完成代码、文献核验、development 阈值操作化、实验执行、失败诊断和分析。
- 用户后续明确授权遇到问题分析设计并重新实验；所有修订均另存、预先固定，旧失败不改。
- 数据为本地 MuJoCo 仿真；没有人类受试者数据或外部模型数据上传。
- 验证状态：工程测试通过；development 证据已分析；正式方法优越性未验证。

## 1. Scientific question 与方法

问题是不同状态下 stochastic search breadth K 与 model-based foresight H 的边际价值，
以及 rollout model 的可靠性是否改变这一分配。不是把两个参数调到某个有利数字。

实现链路：causal observation → scene / ICODE / previous-planner context → continuous
Beta-PPO → integer K/H → 同一 ICODE-corrected MPPI → scan guard → MuJoCo。
预算 policy 不输出机器人动作。I 固定1；没有新增 proposal Actor、learned cost、
IMM、预测模块或 memory。详见[方法文档](../methods/model_reliability_aware_compute_allocation.md)。

## 2. RL formulation

默认41维输入：17个 scene、8个 dynamics descriptor、8个 availability mask、8个历史
planner channel。scene-only17维，scene+dynamics33维，scene+planner25维。
所有当前 ESS/latency/cost 都不进入当前预算状态，raw cost_min 从未作为跨 H 特征。

共享64/64 Tanh MLP，两个独立 Beta heads，alpha/beta=softplus(raw)+1，另有标量 critic。
PPO 使用 clipped surrogate、GAE、entropy、value loss、gradient clipping、running
observation normalization。评估用 Beta mean；checkpoint 保存 optimizer、normalizer、
RNG、动作／观测契约。resume 重新开始 episode，不伪称恢复 MuJoCo 全部隐状态。

原始参数：lr3e-4，gamma.99，lambda.95，128步 rollout，4 epochs，minibatch64，clip.2，
entropy_coef.01，value_coef.5，grad clip.5。没有实现 SAC。

## 3. K/H、warm start、真实计算

先 profile 而后冻结CPU development范围 K128..256、步长32，H10..40、步长1。
规则是在已测矩形内保留 residual P95<=80ms 的最大K，控制周期100ms；512/40的P95
为80.78ms，没有事后放宽80ms阈值。后续发现 K>=128 成功率已饱和，另做预注册的低K
支持区补测，训练范围因此为 **K32..256、步长16，H10..40、步长1**。最大计算上界不变。

连续动作映射为 `min + delta*floor(a*(max-min)/delta+.5)`，明确 half-up rounding。
保存 raw/effective/quantized action 和两种单位的量化误差。
H 先 shift 再 truncate／last-control hold，保持原 blend；K 不清空名义序列。
预分配 sampled-control buffer，study 中为163,840 bytes；原 RandomState 噪声抽样和
独立 rollout 返回值不变。旧默认600/36的包装前后等价测试通过。

## 4. ICODE 的两个角色与实际输入

冻结的 L57 single residual 修正 rollout；独立 causal provider 提供物理 a/B norm、
历史执行控制下 residual norm、normalization support、已完成观测转移 innovation EMA。
T1 各组共用在线三成员 ensemble provider 记录 disagreement／a/B disagreement，只有
配置启用的输入组被 policy 看到。没有读取 pilot 事后 shadow CSV。

support 是启发式，不是校准概率；innovation 也含模型／观测假设的影响。缺失量置0并
带 false mask，reset 时历史信息不可用。所有三个 residual checkpoint 的现时 hash
与研究开始前保存的记录一致，见[完整性核验](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/ensemble_checkpoint_integrity.json)。

训练provenance明确指向`l56_high_dynamic_fixed_plant_v1`：24个training episodes／6053个
transitions，固定L56 plant；训练路线为sweep/chicane/accel-straight。原`unseen` split
主要是reverse-S路线，不应被写成所有物理参数都OOD。当前lower-error L70与该训练plant
一致；higher-error物理默认是不同plant，但已用于本轮development校准，因此也不是本轮
正式hold-out test。旧domain配置中的`long_delay_seen`名称不代表一定被这个L57模型见过。

## 5. Reward 与数值诊断

`rho=T_plan/T_control`；`r_RL=task_scale*r_task-beta*rho-eta*max(0,rho-1)-switch_cost`。
task_scale1、switch0，原 task reward 未重写。K/H/K*H/residual/disagreement 不直接惩罚。
T_plan 覆盖映射、配置到 MPPI 产生动作；actor/context、physics、logging 在边界外，单独记录。

第一轮 beta.1/eta1。原非终止步 task reward中位数.40383、compute penalty.02118。
数值缩放诊断采用 critic内部预测.01*V，初始末层也相应缩放；对外V、GAE及reward保持原单位，
不改变任务／计算的相对效用。最后一组 development 诊断明确采用 beta1/eta1，改变了
开发价格，不假装与.1是同一objective；不跨价格直接比较RL return。

一个重要分析限制：继承的 discounted-potential shaping 在**未折扣的 episode return**
中可能奖励较长驻留。因此报告同时列出success、完成时间和原task return，不能用稍高的
raw return替代更快／更好的任务表现。未借此事后改reward。

## 6. 原 pilot 与 Gate R 失败

旧 pilot：Easy等K*H配置近似等价；Complex balanced进展明显更好但全部未成功；D1/D2
没有建立可靠性对照。旧报告和阈值原样保留。

本轮 Gate R v1 使用相同初态和命令的18个probe，三个事先限定的physical候选都未形成
要求的误差上升。H40 position RMSE：参考.13326m，L70 .11401m，L70+delay .11729m，
delay-only .13944m。ICODE虽改善位置，但一些条件的heading／综合误差变差。

按照用户后续授权，第二版检验低／高速度范围与静止／运动起点。主high/rolling仍失败，
且L70误差.29983m显著低于物理默认.56085m。这揭示了设计错误：**物理默认配置不能直接
叫“matched / reliable”**，增载／低摩擦也不必然让固定名义模型更不准。

## 7. 独立 reliability validation

把前述数据明确用作 calibration，先固定 L70为相对lower-error、物理默认为higher-error，
再生成24条未参与选择的新random probe和新初始状态。未放宽原六项H40阈值。

| H40指标 | Lower / nominal | Higher / nominal | Higher / ICODE |
|---|---:|---:|---:|
| position RMSE (m) | .14378 | .23132 | .17609 |
| heading RMSE (rad) | .13999 | .25473 | .16659 |
| v RMSE (m/s) | .04289 | .06074 | .05391 |
| omega RMSE (rad/s) | .16883 | .24215 | .15750 |
| normalized-state RMSE | .34621 | .56705 | .40941 |

六项判据全部通过：位置上升至少20%且.02m；修正减少至少15%且.01m；综合误差上升至少10%，
修正后不增加。它确认**限定高速度probe分布下的相对误差对照**，不代表exact matching、
不代表每个在线状态都如此，更不等同于H的控制价值已经成立。
所有H1/10/20/40及失败修订见[完整误差表](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/all_gate_errors.csv)。

## 8. Scene gate 与定位混杂

首轮single-obstacle为1/6成功，字面上满足mixed success，但Easy竟失败。诊断显示真实
目标距离1.06–1.24m，而planner的轮式里程计只报约.05m。不能把这种定位失败归因于K/H几何价值。

因此另冻结现有L70/pilot的当前pose/twist传感器模式，隔离wheel-odometry drift。
所有方法共用，未改场景／cost／成功阈值。它提供当前状态，不提供未来truth。
修订后Medium single-obstacle为2/6成功，Easy检查2/2成功，Hard U-trap检查0/2。
这一版本用于后续机制研究；**没有宣称已解决真实里程计鲁棒性**。

## 9. Fixed K/H landscape

先运行108个cell：K128/192/256，H10/20/30/40，Easy/Medium/Hard，三个模型／物理条件，seed0。
再运行96个预先固定的低K／邻近H补测cell；全部共204，未丢失败。
结果支持有限网格内的scene-dependent非单调allocation：Easy广泛成功；Medium有窄成功带；
Hard所有原网格仍失败。K>=128曾掩盖了低预算效应；补测中nominal K32/H20失败，部分较大K成功。

ICODE改善部分较长H或低K配置的可行性，例如Medium H24相对H16的task-return损失在nominal下很大，
在ICODE下接近0。但这不意味着H24比H16更有价值，也没有证明一个单调“可靠性越高就该H越长”的规律。

[所有cell表](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/all_landscape_cells.csv)。
![固定网格](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/figures/E_fixed_landscape.png)
![H的经验对比](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/figures/D_foresight_contrast.png)

## 10. Published baseline fidelity

完整核验：[literature verification](../baselines/literature_verification.md)。

| 对照 | 完成内容 | 不能声称的内容 |
|---|---|---|
| Fixed nominal / ICODE | 原controller及固定修正控制 | 新published baseline或原论文任务复现 |
| MPOPI | [ICRA2023论文](https://arxiv.org/abs/2203.16633) Algorithm1 kernel、full-covariance AIS、作者代码差异核验、共同navigation任务移植 | 原racing／locomotion任务的数值结果已复现 |
| SIS/DTH | [作者说明](https://euncheolim.github.io/Single-Instance-Sampling-for-Real-Time-Task-Space-MPPI-Control/)确认的single-instance sampling + 明示distance-H principle port | 原binary variable-dt网格、Franka完整系统或1kHz速度已复现 |
| RL-Horizon | [Bøhn等](https://torarnj.folk.ntnu.no/lahmpc_nmpc2021_eeb.pdf) learned-H思想的PPO/MPPI移植；同信息H-only对照 | 原SAC／learned terminal value的exact reproduction |
| DM-MPPI | 预印本已核验，未找到足够可靠的可运行作者资产 | 已实现或击败该方法 |

MPOPI paper与当前author-code在非零控制修正项上有差异，均显式实现／测试。
SIS全文精确时间网格仍是fidelity缺口。baseline移植没有包装成原系统的exact reproduction。

## 11. Baseline 开发选择与公平性

先在seed0运行22个预定义candidate×4条件=88个episode。按success优先、mean latency其次
选配置，之后只用seed101做20个配对diagnostic episodes，未再回调设置。
MPOPI nominal/residual都选K32/H16/I2；原moment-AIS的I10/alpha1组合也被测并保留。
SIS principle port选K32，distance-scale3m保持原声明的移植选择。

成本、任务、传感器、actuator、控制周期相同；预测模式nominal/residual分别列出。
这些是development诊断，不是正式8–12seed对比。不能用移植版本击败与否概括原论文系统。

## 12. T0 / T1 training

T0通过。T1为一个training seed0、四个条件（Easy/Medium×两种plant），四种输入／动作消融：
full、scene-only、scene+dynamics、full-information H-only。各轮共同PPO／reward／action范围。
episode round-robin；总transition预算相同，因episode长度不同，各condition曝光数并非完全相同，
已保存在每轮decision文件中。

| 开发设置 | 每方法steps | 导航总steps | T1升级判据 |
|---|---:|---:|---|
| beta.1，原critic单位 | 2048 | 8192 | 全部未通过 |
| beta.1，固定critic scale.01 | 2048 | 8192 | 全部未通过 |
| beta1，scale.01；固定2048/8192评估点 | 8192 | 32768 | 全部未通过 |

共49,152个导航训练transitions。没有继续增加价格、长度或seed。
独立解析bandit同样PPO在8192步把确定性MSE从.08993降至.00511（约94%），说明基本梯度／接口可学习。
这不能证明导航的信用分配、观测充分性和sample efficiency已解决。

[全部PPO更新](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/all_ppo_updates.csv)。
![优化诊断](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/figures/training_diagnostics.png)

## 13. 配对 development 结果

下表均为相同四个case、evaluation seed101，beta1。Latency是episode-balanced mean；
P95列是**各episode P95的平均**，不是pooled P95；所有方法4/4成功、0 collision。
完成时间不含失败删选，因为这组全部成功。没有正式置信区间。

| 方法 | mean latency ms | mean episode P95 ms | 完成时间 s | raw task return |
|---|---:|---:|---:|---:|
| Fixed nominal K128/H16 | **2.114** | **2.279** | 9.175 | 144.911 |
| MPOPI port / nominal | 2.559 | 2.800 | 10.375 | 145.738 |
| SIS-DTH principle / nominal | 2.892 | 3.943 | 11.650 | 147.502 |
| Fixed ICODE K32/H20 | 12.551 | 14.246 | 9.350 | 144.611 |
| H-only，完整信息 | 13.314 | 14.105 | 9.850 | 145.109 |
| MPOPI port / ICODE | 14.450 | 15.316 | 10.325 | 145.722 |
| SIS-DTH principle / ICODE | 15.952 | 22.776 | 11.200 | 146.807 |
| Scene+dynamics KH | 21.030 | 22.971 | **9.125** | 144.894 |
| Full KH | **22.225** | **23.700** | **10.550** | 145.533 |
| Scene-only KH | 22.476 | 24.314 | 9.500 | 144.763 |

完整mean/median/SD/IQR/max及K/H见[CSV](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/development_comparison.csv)
／[JSON](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/development_comparison.json)。SD/IQR描述case间分布，不能当training-seed不确定性。

## 14. Performance–latency Pareto

Full并不在有利的development frontier上：同样4/4成功，mean latency为Fixed nominal的约10.51倍、
MPOPI nominal port的约8.68倍、Fixed ICODE的约1.77倍、H-only的约1.67倍。
它的完成时间也比这些主要对照慢，而不是用更多计算换到更高任务表现。

Success–latency的有限点frontier只有Fixed nominal。按完成时间–latency看，Scene+dynamics
有一个约.05s的小时间优势，但需约10倍nominal latency；不能把这个单seed小差异包装成稳定优势。
Full仍被更简单的点支配。没有足够数据计算可靠的“等latency成功率提升曲线”或做跨点插值承诺。

![A](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/figures/A_performance_latency.png)
![B](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/figures/B_success_p95.png)

## 15. Policy 学到了什么、ICODE是否有独立价值

Full最终mean K137.37、mean H25.15；四个condition的mean K range仅4.20、mean H range仅.61。
初始为固定量化的144/25。它不是max/max或min/min collapse，而是接近初始中间预算、缺少有用分配。
数值拟合改善没有自动变成allocation策略改善。

Scene+dynamics相对scene-only在本组有小幅latency／时间优势，但只有一个training seed，
其变化也未满足T1机制判据。因此**不能claim ICODE reliability input有独立价值**。
Full包含planner信息后也没有更好。没有将某个较好单点换名为proposed来回避失败。

![C](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/figures/C_allocations.png)
![F](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/figures/F_ablation_frontier.png)

## 16. Failure cases 与 qualitative trajectories

- 原可靠性label方向错误；先修正设计并独立验证，没有改旧结果。
- 原高速里程计漂移将模型／几何效应混杂；当前机制数据使用当前状态观测隔离。
- Hard U-trap的原网格全部失败；预算分配没有被证明能解决该backbone的困难规划问题。
- 当前PPO在导航中停留于近中间预算；解析bandit可学习，所以不能简单称为梯度代码bug。
- 预测误差变小并不必然使闭环控制／计算Pareto变好。

下图是一个单独声明的illustrative repeat，不混入上面的统计表；latency-dependent policy
即使同seed也可能有不同K/H，所以不称为旧episode的精确重放。三条路径和原数据均保留。
![实际轨迹](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/figures/qualitative_repeat.png)

## 17. Profiling 与工程验证

最终12-cycle模块profile，residual模式mean/P95贡献（ms）：

| 项目 | Mean | P95 |
|---|---:|---:|
| residual inference，嵌套在rollout | 20.598 | 40.013 |
| nominal derivative，嵌套 | 2.258 | 3.733 |
| batch rollout | 18.421 | 37.263 |
| final rollout | 6.400 | 10.596 |
| sampling/preparation | .207 | .442 |
| cost | .176 | .408 |
| weighting/update | .142 | .157 |
| horizon allocation/copy | .0041 | .0051 |
| action mapping | .0566 | .0856 |
| actor，planner边界外 | .584 | .760 |
| reliability/context，边界外 | 2.219 | 6.228 |

嵌套／边界外时间不能相加成100%。本profile使用未训练Beta产生的预算变化，衡量模块，
不是最终policy benchmark。Residual inference占该profile规划时间约79.45%，仍是瓶颈。
没有改写ICODE核心算法；sample-buffer优化已做等价验证。

环境：Python3.8.10、NumPy1.24.4、PyTorch2.4.1+cpu、MuJoCo3.2.3，单Torch/BLAS线程。
机器有RTX5060 Laptop GPU，但当前研究没有安装／验证CUDA版运行栈。所有latency结论
仅限这个CPU实现，不能冒充GPU最优实现比较。分析阶段另装pandas2.0.3，不改动力学库版本。

最终 **112 tests passed，0 failure/skip**，含动态K/H、warm start、mapping、Beta/PPO数学、
checkpoint、因果性、residual、baseline kernels与MuJoCo。见[JUnit](../../research_artifacts/adaptive_compute_2026-09-07/final_tests.xml)。

## 18. Statistical analysis 与完整性

T2/T3没有获得执行资格，因此没有paired t/Wilcoxon、正式proportion CI或formal effect-size CI。
不能把4个固定case、204个超参数cell、控制步或重叠窗口假装成8–12个独立seed。
正式设计中的四个primary contrasts与Holm校正仍只是计划。

11类解释风险全部检查：Simpson（分scene/condition报告）；ecological（不外推probe均值）；
Berkson（承认development scene筛选）；collider（不删失败）；base-rate（列分母）；
regression-to-mean（校准后独立probe）；survivorship（保留失败）；look-elsewhere（列出全部修订）；
forking paths（另存protocol，不改旧判据）；correlation/causation（不给相关性机制claim）；
reverse causality（只用已完成历史context）。

所有阶段protocol/config hash与冻结记录相符，ICODE三个checkpoint与研究前hash相符。
早期manifest只列active checkpoint，现已用研究前的ensemble manifest补充验证，未改写原manifest；
未来collector也已补齐ensemble/context checkpoint。
绘图脚本一处括号语法错误已修复，仅影响后处理，没有重跑或修改实验数据。图像通过程序QA及视觉检查。

## 19. 未执行的正式部分与fidelity限制

- 没有T2的2–4seed confirmation或T3的8–12seed formal study。
- 没有formal OOD payload/friction/torque组合测试；.10s combined-unseen配置未用于选择。
- 不是所有消融都被训练到qualification；scene+planner、K-only、nominal adaptive等接口可用但无正式结论。
- SIS原binary-time-grid完整复现未完成；DM-MPPI未实现；不能声称完整击败全部原published systems。
- No real-robot／odometry robustness／full-cycle deadline guarantee。
- 一个seed、有限steps没有排除更长训练或其他任务能学会；但没有理由在当前无机制信号时直接扩大昂贵正式实验。

## 20. 八个科学问题的最终回答

| 问题 | 本轮判断 |
|---|---|
| 不同scene是否有不同optimal K/H？ | **development有限网格内支持**不同可行／低latency区域；不是全局最优或正式统计证明 |
| joint是否优于H-only？ | **没有**。当前配对诊断full更贵且未更好 |
| RL是否学到有意义的state allocation？ | **导航中未通过判据**；基本PPO在解析任务可学习，不能将导航负结果简化为代码bug |
| reliability是否改变H价值？ | **部分development证据**：误差对照成立，部分长H可行性恢复；最优foresight价值机制仍未充分建立 |
| ICODE输入优于scene-only？ | **未建立独立、稳定优势** |
| Ours是否在published baselines frontier左上？ | **没有当前证据**；开发移植对照也未被full支配，反而更便宜；原系统／formal superiority未验证 |
| 改善来自performance还是saving？ | **full没有明确改善**；ICODE改善open-loop误差，但学习分配没有兑现成任务／计算优势 |
| 足够支撑ICRA核心claim？ | **NO-GO** |

## 21. 建议与复现入口

保留平台、失败数据和可靠性校准工具。若另开后续研究，应先诊断导航meta-action的信用分配、
PPO随机预算探索对controller的影响，以及真实可观测reliability能否预测**增量闭环效用**，
而不仅是预测误差。先在development证明一个可学习且有用的allocation信号，再重新申请
T2资格；不要直接扩seed、换一套容易阳性的场景或把scene-only小优势包装为reliability贡献。

运行入口位于`experiments/compute_allocation/`：`reliability_gate*.py`、`scene_gate.py`、
`fixed_landscape.py`、`training_screen.py`、`baseline_development.py`、`analyze_study.py`。
每个现有目录的protocol/manifest/resolved YAML是准确复现参数；执行新运行必须用新输出目录。
PPO学习状态保留在各T1 variant的checkpoints目录，hash索引见
[checkpoint hashes](../../research_artifacts/adaptive_compute_2026-09-07/final_analysis/ppo_checkpoint_hashes.json)。

cohort checkpoint 使用专门入口，以便读取并核验原四个条件的完整配置（不与单场景CLI的
metadata契约混用）：

```bash
.venv/bin/python experiments/compute_allocation/evaluate_cohort_checkpoint.py --run-dir research_artifacts/adaptive_compute_2026-09-07/t1_price1 --variant full --verify-only
.venv/bin/python experiments/compute_allocation/evaluate_cohort_checkpoint.py --run-dir research_artifacts/adaptive_compute_2026-09-07/t1_price1 --variant full --seed 101 --output results/adaptive_compute_replay
```

第一条只核验，不执行episode；第二条是可复现入口示例，本报告没有把它当新formal结果。

阶段commit包括ce43b72（连续核心）、c40e7a9（profile）、e46ad82（baseline ports）、
cf39b7d（Gate R冻结）、36c441f/5479e85/60c673e（失败与独立校准）、6532059/dc16ae3（场景隔离／grid）、
a9ea014（低预算补测）、061bb7a（T1）、b061c17/4cc92bb（受控优化诊断）、f1b01c6（bandit／baseline诊断）。
后处理和最终artifacts另作本地commit；没有push、destructive git或覆盖旧实验。
