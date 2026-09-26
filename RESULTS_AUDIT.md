# 结果审计

## Material Passport
- 来源：Windows原型、WSL作者复现源码与raw artifacts，服务器迁移记录。
- 当前核验层次：目录、git、进程及API已直接核查；全文轨迹/模型哈希待服务器 evidence_audit.py。
- 不确定性：历史总结不是完成证明；旧active状态失真；部分历史仿真预算无法仅由manifest恢复。

已存在证据表明原方法重建和多项改进未达到稳定核心效果；暂不将历史报告数值当作本轮重新核算结果。正式结论需审计原始trajectory/model/config/source与所有seed。

最新4个完成标记只证明存在待验证产物。服务器验收结果另写 docs/bohn2021_takeover/server_evidence_audit.json 与 server_smoke.json。工程成功不等于科研成功。

<!-- latency-tree-recovery-migration-amendment-20260926 -->
## 2026-09-26 amendment audit entry

Vehicle selection-noise diagnostics are recorded as development/training-selection evidence only. They support the finding that saved vehicle policies match the reconstructed selection objective but do not support a reproduction or adaptive-horizon success claim. Seed0 and seed1 provide fixed-H/nonadaptive evidence; seed2 is only a candidate. Any future timing claim must be based on AWS-only paired remeasurement. The amendment file and manifest record hashes of the preregistration, migration registration/status, and diagnostic outputs. Validation64 and sealed test128 remained unread during this action.
