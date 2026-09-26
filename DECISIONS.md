# 决策记录

2026-09-26：优先审计 D 盘，但采用 WSL 已存在的作者复现资产作为服务器科研输入。证据：D相关25文件为早期原型；WSL含数十个实验组、多seed模型、原始轨迹和协议。两个工作树保留，既有修改不重置。避免从原型重复昂贵训练。

2026-09-26：复制冻结源码和原始绝对路径兼容链接，不修改已登记源码。新部署文件单列。source SHA、dirty patch 与迁移后的SHA重新登记，不能把新快照提交当原实验提交。

2026-09-26：保留四个完成浅树模型与中断训练，但不静默拼接两台主机的测时。恢复策略先基于源码和未完成边界审计登记；可能需重新完整计量参与排名的配对块，重复预算单列。

2026-09-26：长期控制器只允许gpt-5.5/xhigh，单实验串行，4h wall上限、3.1GB服务内存、48次API/日、每轮6次、150万报告token/日。预算耗尽等待下一个UTC日，不能换模型。连续失败进入诊断。服务器重启由systemd恢复。

2026-09-26追加：服务器已逐文件核验vehicle seed1训练内复选10个条件、160回合、12640控制步，3/4树候选行为始终H25，另1个使用H25/H45且成本更高，最终按冻结规则选择fixed。该保存策略不可能满足逐seed回合内自适应门槛；这不是独立test结论。优先诊断候选退化/覆盖和测时噪声，避免仅完成剩余长训练就误称成功。未读取validation/test结果。



2026-09-26用户覆盖：取消所有API调用、token、API费用日上限，去掉本地单请求输出token上限。仅使用GPT-5.5/xhigh；每轮最多12次调用后保存状态，5秒后继续；保留实验公平预算、单任务执行、实验超时、失败退避、资源限制。旧48次/日与150万token/日规则作废。

<!-- latency-tree-recovery-migration-amendment-20260926 -->
## 2026-09-26 latency-tree recovery/migration amendment

Decision: freeze a recovery/migration amendment before any resumed timing-sensitive latency-tree work. The historical preregistration is preserved unchanged. WSL and AWS wall-time measurements must not be mixed for ranking; any timing-influenced selection requires a whole same-host paired AWS block. Behaviorally fixed trees are fixed-H comparators, not adaptive policies. Adaptive claims now require actually used multiple horizons plus inherited safety/non-inferiority gates and paired cost or same-host timing benefit. The current vehicle training-selection evidence is development-only: seed0 is timing-noise-susceptible fixed-H25 behavior, seed1 is fixed, and only seed2 is a switching candidate requiring independent validation.
