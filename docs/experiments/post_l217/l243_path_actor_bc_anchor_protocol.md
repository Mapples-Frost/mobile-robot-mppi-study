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
   route source 固定为各场景已经审计的 `task.points`，不得重新用 A* 替换；
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

## 6. 离线教师定位契约

直接控制教师跟踪 L222 已审计的 `task.points` 时，显式使用 MuJoCo
ground-truth pose，避免长轨迹轮速里程计漂移把教师带离安全参考线。这是离线教师
专用特权：学生保存的 69 维输入仍逐字节来自 `DirectControlEnv.reset/step` 的正常
observation，不附加真值位姿；在线学生也不读取 simulator truth 或障碍物真值。
manifest 与逐步 audit 必须记录 `teacher_pose_source=ground_truth`。

## 7. 教师开发尝试登记

- v4：`task.points`、perceived pose、lookahead 0.70 m、900 步；因长时轮速
  里程计漂移而偏离参考线，3/3 未到达，冻结保留。
- v5：仅将 teacher pose 改为 ground truth；定位漂移被隔离，但 0.70 m
  前视在首个直角产生约 0.071 m 切弯，最低净空降至约 0.13 m 并被安全链截停，
  3/3 未到达，冻结保留。
- v6：预注册单变量为 lookahead `0.70 -> 0.35 m`；pose source、速度、
  yaw gain、900 步、地图、参考线和安全链均保持 v5 不变。只有 v6 完成后才读取
  结果；不得覆盖 v4/v5。结果为 3/3 零碰撞、零 boundary，平均完成度约
  0.568，末 200 步仍持续前进；未到达由 900 步时限导致。
- v7：预注册单变量为 `max_steps 900 -> 1400`；lookahead 0.35 m、
  ground-truth teacher pose、速度、yaw gain、地图、参考线与安全链保持 v6 不变。
  只有 v7 完成后才读取结果。结果为 3/3 零碰撞、零 boundary，完成度
  0.971--0.975，末 200 步仍前进约 5 m，仍为时限截断。
- v8：预注册单变量为 `max_steps 1400 -> 1500`；其余参数与 v7 完全一致。
  这是根据 v7 剩余 0.69--0.79 m 路径和末段推进率作出的事前时限修正。
  结果为 3/3 成功、零碰撞、零 boundary，学生 observation 为 69 维且不含
  absolute pose。

正式教师数据采集由 `configs/rl/direct_control_bc_teacher_l243.yaml` 的
`collection` 段和 `run_l243_direct_control_teacher_collection.py` 唯一驱动：六张
L222 地图、seeds 20262401--20262415、split seed 20260721、lookahead 0.35 m、
1500 步上限均在运行前冻结。采集器不允许空 split；任一 split 没有成功教师回合即
fail closed。

首次 90 回合采集在写最终 manifest 时暴露了 v3 多场景计数校验缺陷：旧校验器只
计 15 个 split seeds，未乘六个 scene。全部 MuJoCo 轨迹、resolved config 和 split
shards 已在异常前完整落盘，禁止浪费性重跑。修复校验器后，由
`finalize_l243_teacher_dataset.py` 对 90 个 scene×seed 键、shard episode IDs、哈希和
69D student contract 进行 fail-closed 重建；manifest 同时记录原采集 Git SHA 与最终
校验 Git SHA。

完整数据集包含 90 个回合，全部零碰撞、零 boundary；Serpentine、Giant-U、
Nested-U、Cylinder-Spiral 共 60 个成功回合进入 student shards，Opposed-U 与
Cylinder-Forest 的 30 个失败回合只保留在 privileged audit。train/validation/test
分别包含 36/12/12 个成功 episode 和 25159/8390/8383 个样本。该结果不做筛选：
BC anchor 只在有安全成功教师证据的四图提供稳定化，SAC 仍在六图环境交互中学习。

冻结 BC 配置为 batch size 256、mean weight 2.0、log-std weight 0.001、target
log-std -2.0。完整候选使用 seeds 20262331--33 和 validation bases
20262931--33；仅 smoke 配置把 actor update 起点改为 8，以在 60 步内验证 BC
更新链路，完整 30k 配置仍保持 actor update after 3000。

BC smoke 首次启动在更新前被旧 actor-initialization contract 拦截：该检查错误地要求
源 checkpoint 与新训练使用相同 BC dataset。L243 只执行 actor-only warm start，不继承
optimizer、replay 或训练目标，因此允许目标 run 新增 BC anchor；严格 dataset fingerprint
匹配仍保留在 resume 路径。修复后初始化 provenance 同时记录 source/target BC fingerprint。

第二次 smoke 在更新前触发既有的 normalizer 防漂移约束：BC anchor 训练必须冻结
初始化 checkpoint 已拟合的 observation normalizer，否则固定教师样本的归一化坐标会
随在线数据变化。L243 因此预注册 `normalizer_update=frozen`；这不会把真值加入学生
输入，也不会冻结 Actor、critic 或 SAC 更新。

第三次 smoke 通过：60 environment steps 产生 53 条 update records（step 8--60），
`bc_anchor_mean_mse`、RMSE 与 log-std 项均为有限值，action mode 为 direct-control，
observation/action dimensions 为 69/2，无 NaN/Inf。前两次启动失败作为工程负向记录
保留，不计作训练效果。
