# L219 六地图 residual-conditioned Actor 开发 Gate 结果

## 结论

L219 首轮 Development Gate **未通过**，因此没有创建 sealed manifest，也没有使用任何 sealed seed。该负向结果完整保留。

积极结果是：新 Actor 在六地图上获得了非零 support/proposal authority；Full Proposed 相对无门控的 Simple combination 在相同 rollout 预算下明显更稳，且三种新方法均未发生碰撞。这说明 role-aware HSS 确实阻止了部分危险 proposal。

失败点是：新 Full Proposed 相对 ICODE-MPPI 的成功率下降 `0.2222`，平均路径完成度下降 `0.1761`；相对旧 Full 的方向相近。原本成功的 opposed-U 从 `3/3` 降为 `0/3`，cylinder-forest 从 `3/3` 降为 `2/3`；三张旧失败地图没有达到预先要求的成功率提升或 `+0.10` 完成度改善。

## 实验范围与来源

- 方法：ICODE-MPPI、Simple combination、新 Actor Full Proposed；
- 场景：serpentine、giant-U、opposed-U、nested-U、cylinder forest、cylinder spiral；
- 独立实验单位：seed `91001--91003`；六场景是 seed 内重复 strata；
- 新实验回合：`54 = 3 methods × 6 scenes × 3 seeds`；
- 旧 Full 辅助回合：18，仅作有来源历史对照，不冒充同轮随机化方法；
- 后端：MuJoCo `3.2.3`；
- qualification：全部为 1；
- rollout 预算：每次决策 30，1 iteration；
- 代码：Git `28378a60548a35768b4222723914921ab941c2a2`；
- Actor SHA-256：`bf26a67ebac313930d63760db931e5d50704cdd9181923afd7f66d16159356b4`；
- 推断：10,000 次 seed-cluster bootstrap；只有三个开发 seed，不作为论文正式统计结论。

## 关键描述统计

| 场景 | ICODE success | Simple success | New Full success | Old Full success | New Full proposal authority |
|---|---:|---:|---:|---:|---:|
| serpentine | 0/3 | 0/3 | 0/3 | 0/3 | 0.1824 |
| giant-U | 3/3 | 0/3 | 3/3 | 3/3 | 0.7256 |
| opposed-U | 3/3 | 0/3 | 0/3 | 3/3 | 0.9725 |
| nested-U | 0/3 | 0/3 | 0/3 | 0/3 | 0.0000 |
| cylinder forest | 3/3 | 0/3 | 2/3 | 3/3 | 0.3116 |
| cylinder spiral | 0/3 | 0/3 | 0/3 | 0/3 | 0.0000 |

## 工程根因

逐回合审计发现，`reliability_actor_competence_*` 始终为 1，而 `guided_yield` 与 `gaussian_yield` 始终为 0。代码已经实现了按 Actor-guided/高斯候选的机会归一化精英率更新 competence，但继承配置中的 `source_competence_enabled` 为 false。

因此，HSS 只依据离线 support 与 ICODE reliability 分配权限，没有使用当前场景中“Actor 候选实际上有没有成为优质 MPPI 候选”的在线证据。opposed-U 的 Actor proposal 明显无效，却仍取得约 97% 平均 proposal authority。这是角色感知门控未完整启用，而不是修改论文核心机制的理由。

## 下一步

L220 只启用已有的 causal source-relative competence，保持 ICODE、Actor checkpoint、MPPI、场景、预算和安全链不变。先用 development seed 做三场景机制探针；只有消除 opposed-U/forest 回退且 competence 日志真实更新，才继续六地图完整 Gate。若仍不能改善旧失败地图，再在相同核心机制下把 RL prior 改为对可信 warm-start 的有界修正，并单独预注册该开发候选。

