# Bøhn 2021 复现证据地图与本轮接续

## Material Passport

- 用户目标：可靠重现学习自适应预测时域相对于合理固定 H 的控制性能与计算代价权衡；优先车辆、倒立摆。
- 来源：论文、固定作者历史源码、本工作区保存的模型、原始轨迹和协议。
- 当前结论：尚未得到两任务跨种子稳定优势；不得标记复现成功。
- 本轮继续原工作区，不开展移动机器人联合 K/H，不重新训练已有完成模型。

## 文件与审计核验

2026-09-24 本轮接续时没有旧训练进程。分支校准目录已有协议、输入散列、十个场景库和一个完成的车辆冒烟；没有六组正式校准模型。已重新运行 `branch_calibration_audit.py smoke --replay`，通过物理成本、观测、回报、参数冻结、序列回放和重载一致性核验。因此复用了该冒烟，没有重复执行拟合。

原方法当前论文配置的 `paper_grid_audit_report.py` 重新执行退出码 0，核验 26 模型的 520 个验证条件和 1,040 个旧保留评估条件。`min_q_audit_report.py --split validation` 重新执行退出码 0。两个日志保存在 `results/campaign_inventory_2026-09-24/`。旧保留集已经暴露，本轮不把它们当作新的独立证据。

`campaign_inventory.py` 文件清点得到 168 份符合作者训练格式的运行 manifest，跨 18 个目录组；不是全项目训练预算。完整模型、未完成已刷盘步数、归档、参数散列及其他统计见对应 `inventory.json`。不同格式的 PyTorch 扩展、额外分支、reset、未刷盘和失败工作不能从这个小计中消失，须另外披露。

## 已有证据的边界

|问题|已有结果|可以作出的判断|不能作出的判断|
|---|---|---|---|
|是否找回原实现|找回作者同期 SAC/AHMPC/do-mpc 分支，保留固定提交和许可证|核心方法可追溯|原完整配置、测试文件和精确数字已恢复|
|原方法重建|当前完整固定 H 网格及三种子 RL 未给出稳定优势|当前重建未复现核心效果|原论文错误或所有自适应 H 无效|
|是不是不可恢复初值的唯一影响|历史收窄初始扰动后 RL 仍落后；当前较宽分布存在共同失败|初值难度不是唯一原因|可删掉失败场景以得到优势|
|终端价值是否有效|特定近直立实验清零终端值显著恶化；当前终端移植验证失败|终端值在部分条件有作用，移植不是稳定修复|终端学习足以保证自适应 H 优势|
|actor 是否充分利用目标|当前 48 个验证状态中 43 个存在数值判据支持的同标准差更高目标候选|局部目标利用有改进空间|共享网络一定有实现 bug，或更高 Q 保证更好控制|
|critic 直接推荐能否修复|确定性及匹配 soft-return 续跑均存在反例|不能直接把 Q 贪心当作可靠修复|两个场景足以估计总体错误率|
|min-Q、低熵、统一裁剪是否有效|各自保留负结果和种子敏感性；统一裁剪出现严重求解退化|这些已试改动未建立通用稳定效果|挑一个胜出种子即可宣布成功|
|计算代价|原实现固定最大 50 步 NLP，通过 hend 屏蔽；H 为线性代理|必须单独测实际耗时|平均 H 较小就等于实际加速|

主要报告：`bohn2021_handoff_2026-09-18.md`、`bohn2021_min_q_extension_2026-09-24.md`、`bohn2021_paper_grid_validation_diagnosis_2026-09-24.md`、`bohn2021_terminal_transplant_2026-09-24.md`、`bohn2021_paper_h_credit_2026-09-24.md`、`bohn2021_paper_h_soft_2026-09-24.md`。各报告对应不同场景库和任务变体，不能跨表直接按成本绝对值排名。

## 本轮采用的有限改进实验

采用此前已冻结、尚未正式执行的分支回报校准，因为它直接针对当前 critic 排序与实测续控不一致的证据。不是假定该路线必然成功，也不把训练拟合成功当作闭环控制成功。

- 两任务全部训练种子 0/1/2；复用已完成 15,000 步的 min-Q 模型。
- 新训练场景每模型 8 个；每个可用场景在第 20、60 步建立状态锚点。结束过早的场景保留原因，不补抽。
- 候选 H 为 1、10、20、25、30、40、50，加原 actor 的当前 H；每个 H 两次共享噪声续跑。
- 首步动作强制，随后恢复原随机 actor；按 gamma=.97、原奖励除数、后续熵与时间截断 target-V 计算回报。
- 只拟合 Q1/Q2 的状态内中心化回报，固定 1,000 次更新；actor、V、target-V 和 MPC 终端多项式不变。
- 闭环比较 actor、未校准 Q 贪心、校准 Q 贪心和独立训练固定 H，全部 24 个条件使用新的每任务 10 场景验证库。
- 固定 H 来自历史完整十值搜索，各候选独立训练终端价值；预选车辆 H25、倒立摆 H30 另补齐种子 1/2。公开 300,000 搜索步、60,000 补充固定步，以及原 actor/min-Q 各 90,000 步；不声称额外分支校准与固定方法等仿真预算。
- 验证门槛、独立测试门槛以 `branch_calibration_2026-09-24/protocol.json` 为准。所有拟合与 24 条件审计完整且验证门槛通过后，才开启每任务 20 场景的新测试库；验证失败则封存不动。

十个当前场景库之间的场景散列没有完全重复；此检查不读取测试控制结果。不能把场景库生成或 hash 检查叫做测试性能评估。

在任何新验证轨迹生成之前，另存 `claim_requirements.json`，说明用户层面的成功结论还须满足每个独立测试种子的任务成功次数不低于全部三个对照。原门槛只将成本、约束及求解失败计入布尔判定，虽报告 goal 数却未将其作为门槛；因此原门槛通过只是必要条件，不能掩盖任务完成率下降。该补充不放宽原标准、不修改训练、不用于重新选模型。独立加速结论另要求全部独立测试场景串行实测，目前安排的验证计时仅作描述。

## 实测计算代价补充

独立协议 `serial_timing/protocol.json` 在本轮正式验证结果产生之前登记。所有四方法、两任务、三种子和全部十个验证场景各串行重复两次，条件顺序预设随机化。分别计时策略选择与 `controller.get_action`，主耗时为二者之和；该控制器范围包含 NLP 求解、预测提取和内部记录，不包含物理仿真、奖励核算、审计和磁盘写入。全环境步耗时与 reset 另存。

每次计时的完整轨迹须与已审计的验证轨迹逐项一致；失败条件照常计入。环境版本、CPU、线程配置、原始逐步耗时和进程冲突检查保留。WSL 调度和未控制的宿主活动仍是限制；本次验证计时本身不建立独立加速结论。

本轮已完成包装技术检查：1 次重置预热和 3 个烟测步，逐状态、动作、成本精确回放一致。该检查并行于训练，因此不报告它的速度数值；额外仿真预算 1 次 reset、3 步另记。

## 接续命令与状态入口

在 WSL 项目根目录执行：

```bash
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_status.py
PY=/home/mapples/.local/share/bohn2021-python37/bin/python
# 六组已完成，不重复训练；统一使用修正浮点运算顺序的审计
$PY experiments/bohn2021_reproduction/branch_calibration_resume.py audit --phase training
$PY experiments/bohn2021_reproduction/branch_calibration_run.py evaluate --split validation
$PY experiments/bohn2021_reproduction/branch_calibration_audit.py validation
$PY experiments/bohn2021_reproduction/branch_calibration_diagnosis.py
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_delivery.py
# 无其他项目仿真进程时执行；该步骤不控制独立测试解封
$PY experiments/bohn2021_reproduction/branch_calibration_timing.py
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_timing_report.py
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_failure_diagnosis.py
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_claim_audit.py
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_delivery.py
```

`branch_calibration_status.py` 以文件和存活 PID 显示实时状态。此文不是完成标记。未完成拟合、未运行测试或未完成计时不得写成已完成；Goal 保持 active。

## 数值审计修正记录

车辆 seed 0 在保存 `case05_t020.json` 后因审计中断，已保留失败文件及 14,134 次步调用。失败来自独立审计的对数 Jacobian 舍入顺序：原 TF1 图优化为 `float32(1+EPS)-float32(a*a)`，旧 NumPy 审计是 `(float32(1)-float32(a*a))+EPS`。动作接近 ±1 时，两者分母相差约一个浮点单位，被对数放大，首条差值约 4.03e-5，略超 4e-5 容差。

两条问题记录在原冻结网络、原观测、原噪声下重算，与保存的动作参数及 log probability 逐值相同。`numeric_failure_diagnosis.json` 保存了证据。新独立审计只修正运算顺序，没有放宽 4e-5 容差；当时已落盘的两个车辆种子共 344 条分支重审通过。

`numeric_amendment.json` 冻结补丁代码和证据散列；原协议、所有原源码、轨迹、标签、拟合和效果验收均不修改。`branch_calibration_resume.py` 通过明确的运行时审计替换恢复缺失工作，保留原队列中其他任务。原队列结束后执行：

```bash
# 只恢复未完成组；对已完成组调用也仅重审，不重新训练
$PY experiments/bohn2021_reproduction/branch_calibration_resume.py train --task vehicle --seed 0
# 全部六组应统一使用修正后的审计（替代上面的原 training 审计命令）
$PY experiments/bohn2021_reproduction/branch_calibration_resume.py audit --phase training
```

`smoke_audit.log` 保留最初精确回放通过的记录；后续 `numeric_smoke_audit.log` 是不新增仿真的数值重审，当前 `smoke_audit.json` 对应该次重审。两者不能重复计为新增训练。

原队列随后在倒立摆 seed2 也触发相同审计失败，退出时六组返回码为 `[1,0,0,0,0,1]`。其已保存 16 条分支经修正审计全部通过，九条超出旧审计容差的记录用原网络重放完全一致，证据在 `numeric_pendulum_s2_check.json`。原始失败日志继续保留。四个成功模型不重新训练，恢复流程分别补齐车辆 seed0 和倒立摆 seed2 的缺失部分。

`branch_calibration_claim_audit.py` 检查原门槛与额外成功率要求，只有已存在且已审计的独立测试结果才进入对应表格；它不会为填表读取未解封测试场景，也不会自动把控制效果通过当成整个 Goal 完成。

## 正式训练阶段完成记录

六组拟合现已全部完成，`training_audit.json` 的 `passed=true`，全部 source/branch 轨迹、观测、成本、回报、参数冻结、模型重载、文件散列和拟合目标均核验通过。数值补丁与审计的关联记录为 `numeric_training_audit_provenance.json`。

统一采集预算：48 个原策略场景回合共 3,802 步；前缀重放 50,180 步；1,202 条后缀共 61,671 步；合计 115,653 次显式 step，1,327 次显式 reset；共 77 个可用锚点，19 个因提前结束而跳过的锚点。六组各 1,000 次监督更新，共 6,000 次。原始失败发生在分支写盘后，因此恢复复用这些完整产物；审计的全部尝试 step 总数恰等于上述采集计数，没有重复训练。构造器、场景生成、冒烟、独立回放、计时包装检查等开销另外保留，不把 115,653 称为生命周期总预算。

完整 24 条件 × 每条件 10 场景的验证现已完成，共 240 回合、20,935 步，数据审计通过；预注册效果门槛失败。两个任务的三个种子中，校准策略的平均成本全部高于未改动的 min-Q actor 和独立训练固定 H。车辆校准成功次数为 7/6/9（每种子十场景），固定 H 为 10/10/10；倒立摆各方法各种子均有一个越界回合。不能将训练拟合改善当作控制改善。

车辆对 raw-Q 贪心的两种子改善条件通过，但 seed1 大幅恶化；倒立摆对 raw-Q 的条件也未通过。完整原始轨迹、所有种子和失败都保留。未生成 `evaluations/test`，独立测试继续封存。

串行计时已完成 48/48 条件、480 个回合重复、41,870 步，全部轨迹精确回放一致；`serial_timing/timing_audit.json` 通过。校准相对固定 H 的每步平均耗时比：车辆 1.595/3.838/1.263，倒立摆 0.971/1.117/1.061。两次重复均值的最大/最小比最高约 1.198，不把小差异当作可靠加速。完整失败诊断完成 240 回合和 180 个配对差，结论审计明确 `core_reproduction_achieved=false`。本轮所有流程退出码 0，没有训练或计时任务遗留；效果门槛仍为失败。

最终阶段证据与剩余路线见 [失败评估](bohn2021_failure_assessment_2026-09-24.md)，原始结果、模型、命令、环境、CSV 和 PNG/PDF 见本轮 `delivery/report_CN.md`。当前 Goal 未标记成功或完成。

本轮 `actor` 表示 min-Q 改进模型的未改动 actor，不是最初作者 Q1 actor 原方法。图表已作相应标识；原方法与本轮改进结果需分别阅读，不能把不同场景库的成本直接拼表排名。
