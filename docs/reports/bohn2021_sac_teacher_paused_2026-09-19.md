# SAC教师对照：关机暂停与续跑入口

> **2026-09-19 12:27 已续跑完成，本文不再是待办入口。** [最终报告](bohn2021_sac_teacher_2026-09-19.md)。6个正式模型、116次验证和144次holdout均完成并审计通过；教师辅助平均450.7181，普通21k为270.0414，固定H25为141.9593，三个教师种子均劣于同预算普通组。旧中断轨迹19,348条与重跑完全一致，旧检查点权重相同；seed2模型逐文件未改变。当前没有待恢复训练。后续建议见最终报告，不重复运行本页历史恢复步骤。

> 2026-09-19 11:26（Asia/Shanghai）用户在新会话要求接续，已恢复执行。使用独立 `sac_teacher_recovery.py` 校验冻结散列，将 plain_s0/teacher_s1 完整移入 `interrupted_shutdown/`，保留逐文件散列后按原种子重跑。已完成 seed2 两模型跳过训练。恢复清单 `recovery.json`；最终核验 `recovery_audit.json` 将比较旧检查点与重跑结果。下文为暂停时的历史快照，不表示当前进程状态。

用户于2026-09-19 01:05（Asia/Shanghai）要求停止，明天继续。已冻结并终止本轮dispatcher及两个训练worker，验证进程全部退出。没有创建自动唤醒或后台续跑任务。

## 磁盘状态

根目录：`research_artifacts/bohn2021_reproduction_2026-09-17/results/sac_teacher`。

|模型|状态|保存的完整转移|最近完整检查点|
|---|---|---:|---|
|plain_s2|完成21k在线步/20,901更新|21,000|step_21000.zip及state.pkl|
|teacher_s2|完成6k教师+15k在线/20,901更新|21,000|step_15000.zip及state.pkl|
|plain_s0|用户暂停|9,045|step_05000.zip及state.pkl|
|teacher_s1|用户暂停|10,303（教师6,000+在线4,303）|pretrained.zip及state.pkl|
|plain_s1|未启动|0|无|
|teacher_s0|正式训练未启动|0|仅另有独立smoke，不算正式模型|

两个中断日志均没有截断JSON行。实际执行但尚未刷盘的最后少量转移不能从日志精确计数，暂停时progress记录与磁盘完整行数均保留。原始文件不删除、不覆盖。

## 明天的执行顺序

1. 先读 `paused_shutdown.json`、`protocol.json`、`training_audit.json` 和本文件。用户说继续即恢复授权，不需再次询问。
2. 不直接无脑重跑suite：`train()`会保护部分目录，遇到已有transitions.jsonl时断言停止。
3. 首选可审计恢复：把两个部分运行移到明确的 `interrupted_shutdown/` 子目录，保持所有检查点和日志；然后按原种子从头重跑这两个未完成模型，核对已有5k/预训练检查点的一致性，额外仿真预算单列。已完成seed2绝不能重训。仅在完成TF随机流和MPC状态恢复验证后才宣称精确断点恢复；现有pickle不支持这样的声明。
4. `sac_teacher_study.py`、其依赖、配置和场景在 `hashes.json` 冻结。恢复方案应写独立文件/附录，不暗改冻结源码。新增report和diagnostic脚本未纳入训练散列，可修正报告错误。
5. 使用旧Python：`/home/mapples/.local/share/bohn2021-python37/bin/python -u experiments/bohn2021_reproduction/sac_teacher_study.py`。suite会跳过completed模型，启动剩余训练，再自动执行验证粗细固定H选择和新holdout评价。最多2个MPC进程。
6. 完成训练后运行 `sac_teacher_diagnostic.py`（旧48验证锚点，只作冻结诊断，不是soft-Q真值）。完成全部评价后运行 `sac_teacher_report.py`，修正若有报告实现问题、核对预算、目视检查PNG/PDF。
7. 报告生成脚本尚未跑过全套结果。`audit_all()`要求6个完整模型；需加计中断重跑预算。最终报告目标 `docs/reports/bohn2021_sac_teacher_2026-09-19.md` 尚未生成，不能引用为已有结果。
8. 最后更新handoff、artifact报告索引与ARA会话，保留所有种子/失败。尚无验证或holdout性能结果，不能根据当前训练日志宣称教师辅助SAC成功。

## 已完成核验与发现

- smoke256教师步+300在线步、211次SAC更新，独立物理/回放审计通过；无物理失败，但11个求解失败步。
- 完成的seed2两模型初始权重哈希相同，各21,000转移/20,901更新；全部replay逐项验证了obs/action/reward/next_obs/done。原reward只除以.6一次，600步末端不自举。
- plain_s2训练37次reset、2次物理约束终止、2,022个求解失败步；teacher_s2含采集共35次reset、0次物理终止、2,418个求解失败步。两者是随机探索训练统计，不是冻结评价结论。
- 训练求解失败集中在H2/H3，观测到较大优化约束残差。原环境仍执行求解器返回的控制；不能只报物理失败率，也不能通过事后删样本制造成功。
- 完成两模型及smoke保存/重载权重完全一致。作者3个源码库仍为原固定提交且干净。
- 主要比较：普通21k与教师6k+在线15k，更新总数相同；普通15k为次要在线预算对照。教师为80%因果H5/H30规则+20%连续随机H，没有旧监督模型、actor模仿或教师Q硬标签。
- 此轮仅冻结终端的SAC机制实验，尚未联合学习终端，非原论文性能复现。完整预告56维、新平台场景、alpha=.01、有限回合末端处理与自定义采样循环差异已登记。

协议：`experiments/bohn2021_reproduction/SAC_TEACHER_PROTOCOL.md`。暂停证据：`results/sac_teacher/paused_shutdown.json`。训练审计：`training_audit.json`；重载审计：`reload_audit.json`；源码审计：`environment_audit.json`。不进行holdout反馈调参。
