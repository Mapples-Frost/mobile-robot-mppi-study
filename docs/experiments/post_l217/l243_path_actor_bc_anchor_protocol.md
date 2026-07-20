# L243 Path Actor 的安全参考 BC Anchor 开发协议

日期：2026-07-21  
状态：预注册 development protocol，尚未执行 L243 outcome

## 1. 触发原因

L241 的三个 30k path-conditioned SAC 候选均完成真实更新，但冻结 validation
选择回到 step 0；L242 中该初始化策略虽然被 HSS 非平凡使用，却未改善 Tracking
完成度。继续调 MPPI、地图或安全距离不能解决 Actor 没学会路径条件的问题。

## 2. 单一修复因素

仅启用已有 trainer 的 `behavior_cloning_anchor`，教师来自相同 safe-reference
path 的传统路径跟踪控制器成功转移；SAC、观测、reward、训练场景、ICODE、HSS、
MPPI、MuJoCo 与安全链保持不变。

训练目标为：

```text
L_actor = L_SAC + lambda_BC * L_path_teacher
```

BC 只作为 Actor 优化的稳定化项，critic、entropy 和环境交互仍由 SAC 学习；不得
把 L243 写成纯监督学习。教师数据不得来自 L234 development outcome 或 sealed
seeds，不得包含 simulator obstacle truth 形式的在线 planner 输入。

## 3. 数据与隔离

1. 先从 L222 六张 safe-reference 训练地图生成版本化教师数据；正式教师 seeds
   固定为 `20262401--20262415`，split seed 固定为 `20260721`；
2. 只保留零碰撞、无 boundary violation 的教师片段；所有丢弃原因写入 manifest；
3. 教师生成 seed 与训练 seeds `20262331--20262333`、validation bases
   `20262931--20262933`、development seed `923301001` 相互隔离；
4. 数据记录 scene、physics、seed、Git SHA、config SHA、trajectory 和 teacher
   action；
5. 三候选各训练 30,000 environment steps，初始化仍为 L219 frozen Actor 的
   69 维零列扩展版本。

L241 Actor 的 action contract 是 normalized direct control `(v, omega)`，不能把
仓库已有的 local-subgoal `(distance, bearing)` 教师标签直接当作同一种 action。
采集前必须新增版本化的 `normalized_direct_control` demonstration schema、loader
校验和单元测试；BC anchor 只有在 dataset action contract 与 checkpoint policy
mode 都为 `direct` 时才允许启用，否则 fail closed。

## 4. 冻结选择与 Gate

候选仍按 L241 完全相同的 validation 字典序选择：碰撞、成功、完成度、cross-track
RMSE、goal distance、return、seed/step tie-break。不得根据 L234 outcome 改选。

选中后才允许在 development seed `923301001` 上与：

- ICODE-MPPI；
- L239 old Actor Full；
- L242 selected-initial Full；

进行三个 Tracking 场景的等 rollout 预算比较。

## 5. 通过条件

1. 训练工程完整、update records > 0、无 NaN/Inf；
2. 冻结选择不得回到 global step 0；
3. validation 碰撞不增加，成功数不低于 L241 selected initial；
4. L234 三场景零碰撞；
5. 相对 L242 Full 平均 completion 提升至少 0.02，且无场景回退超过 0.02；
6. proposal authority 与 guided elites 非平凡；
7. 同时报告计算开销。

未通过则保留全部负向结果并停止 Tracking sealed 注册。不得筛 seed、删除失败、
放松 scan_guard 或改变论文核心机制。
