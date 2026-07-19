# L185 独立确认几何冻结修订

日期：2026-07-19  
状态：在开发种子 551–555 和确认种子 561–565 运行前冻结  
上游预注册：`docs/rl/185_path_conditioned_actor_development_prereg_2026-07-19.md`

## 时间顺序与透明说明

L185 原预注册已经冻结确认种子 561–565，并规定确认必须使用未参与训练/
开发的新路径，但当时尚未把路径坐标写入文件。现补齐这一机械性缺口。

本修订发生时，训练种子 20261901 的 10k internal-validation block 已经生成；
该 block 只包含预注册的 reverse-S 与 hairpin，不能用于修改 L185 网络、
reward、观测尺度、训练路径或开发 Gate。开发种子 551–555 尚未运行，本文
定义的两条确认路径也尚未执行。披露这一顺序是为了避免把本修订误称为完全
盲化的原始预注册。

## 冻结确认路径

1. `configs/research/mujoco_path_offset_serpentine_holdout_l186.yaml`；
2. `configs/research/mujoco_path_asymmetric_hairpin_holdout_l186.yaml`。

两条路径均：

- 不含障碍物，专门隔离路径动力学与控制问题；
- 不进入 SAC replay、internal validation 或 551–555 开发评价；
- 不基于 403–407 或 551–555 的结果设计；
- 使用与高动态 differential-drive benchmark 相同的 MuJoCo plant；
- 在确认数据打开后不再修改坐标、lookahead 或 max_steps。

确认物理域固定为：

1. `nominal_seen`；
2. `long_delay_seen`；
3. `combined_unseen`。

确认 episode 数为：

\[
2\ \text{paths}\times 3\ \text{domains}\times 5\ \text{seeds}
=30
\]

每个比较方法均使用同一组 30 个 paired blocks。其余主终点、Gate、cluster
bootstrap 单位和结论边界继续遵循 L185，不因本修订改变。
