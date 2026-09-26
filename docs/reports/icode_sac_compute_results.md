# ICODE 与联合 K/H：持续实验记录

本报告由实验目录自动生成。研究仍在开发阶段；旧 v1/v2 与先前 3216 回合探索的结论保持冻结。所有当前结果均为 MuJoCo，使用理想位姿/速度观测与模拟激光，未验证真实机器人或里程计噪声鲁棒性。

## 实验量

| 阶段 | 完整回合 | 控制周期 |
|---|---|---|
| icode_domain_screen | 288 | 42583 |
| ind_icode_H_full_s9091301 | 420 | 60066 |
| ind_icode_H_full_s9091302 | 417 | 60017 |
| ind_icode_KH_full_s9091301 | 411 | 60165 |
| ind_icode_KH_full_s9091302 | 427 | 60084 |
| ind_icode_KH_masked_s9091301 | 413 | 60156 |
| ind_icode_KH_masked_s9091302 | 418 | 60113 |
| ind_icode_K_full_s9091301 | 423 | 60076 |
| ind_icode_K_full_s9091302 | 429 | 60120 |
| independent_validation_2026-09-09 | 624 | 88413 |
| mechanism_validation_2026-09-09 | 960 | 142880 |
| model_screen | 288 | 54323 |
| obs_icode_H_full_s9091201 | 423 | 60224 |
| obs_icode_KH_full_s9091201 | 414 | 60096 |
| obs_icode_KH_masked_s9091201 | 411 | 60045 |
| obs_icode_K_full_s9091201 | 423 | 60228 |
| observable_validation | 1200 | 171789 |
| sac_icode_H_full_s9091001 | 215 | 40227 |
| sac_icode_KH_full_s9091001 | 200 | 40017 |
| sac_icode_KH_masked_s9091001 | 216 | 40022 |
| sac_icode_K_full_s9091001 | 227 | 40127 |
| shared_backbone_budget_check | 432 | 73943 |
| shared_backbone_factorial | 192 | 34609 |
| validation_4 | 1152 | 216180 |

当前轮已完成阶段合计11023回合、1706503控制周期。控制周期并非独立统计样本。

## 训练诊断

下表为随机探索策略在训练过程中的实际轨迹。场景分布、检查点与回合长度会变化，不能用于配对方法优劣判断。

| 训练组 | 回合 | 周期 | 成功 | 碰撞 | 平均Q | 计算秒/回合 |
|---|---|---|---|---|---|---|
| ind_icode_H_full_s9091301 | 420 | 60066 | 409 | 9 | 0.858 | 3.613 |
| ind_icode_H_full_s9091302 | 417 | 60017 | 400 | 9 | 0.921 | 2.783 |
| ind_icode_KH_full_s9091301 | 411 | 60165 | 378 | 12 | 1.136 | 3.077 |
| ind_icode_KH_full_s9091302 | 427 | 60084 | 412 | 13 | 0.951 | 3.464 |
| ind_icode_KH_masked_s9091301 | 413 | 60156 | 385 | 10 | 1.053 | 3.070 |
| ind_icode_KH_masked_s9091302 | 418 | 60113 | 400 | 8 | 0.904 | 3.400 |
| ind_icode_K_full_s9091301 | 423 | 60076 | 414 | 6 | 0.784 | 4.030 |
| ind_icode_K_full_s9091302 | 429 | 60120 | 416 | 10 | 0.878 | 3.875 |
| obs_icode_H_full_s9091201 | 423 | 60224 | 410 | 8 | 0.853 | 3.394 |
| obs_icode_KH_full_s9091201 | 414 | 60096 | 389 | 11 | 1.023 | 3.690 |
| obs_icode_KH_masked_s9091201 | 411 | 60045 | 392 | 7 | 0.904 | 3.339 |
| obs_icode_K_full_s9091201 | 423 | 60228 | 405 | 12 | 0.967 | 3.813 |
| sac_icode_H_full_s9091001 | 215 | 40227 | 130 | 16 | 2.895 | 4.414 |
| sac_icode_KH_full_s9091001 | 200 | 40017 | 106 | 12 | 3.159 | 3.520 |
| sac_icode_KH_masked_s9091001 | 216 | 40022 | 134 | 12 | 2.686 | 3.654 |
| sac_icode_K_full_s9091001 | 227 | 40127 | 155 | 15 | 2.429 | 5.342 |

## icode_domain_screen

| 方法 | n | 成功 | 碰撞 | 平均Q | 计算秒/回合 | 平均决策ms | deadline miss |
|---|---|---|---|---|---|---|---|
| current_9kg_icode_K128_H32 | 24 | 21 | 2 | 1.668 | 4.265 | 28.54 | 0 |
| current_9kg_icode_K16_H20 | 24 | 18 | 2 | 2.192 | 1.921 | 12.06 | 0 |
| current_9kg_icode_K64_H24 | 24 | 22 | 1 | 1.248 | 2.645 | 17.59 | 0 |
| current_9kg_nominal_K128_H32 | 24 | 21 | 2 | 1.693 | 0.812 | 5.47 | 0 |
| current_9kg_nominal_K16_H20 | 24 | 20 | 1 | 1.582 | 0.590 | 3.66 | 0 |
| current_9kg_nominal_K64_H24 | 24 | 22 | 2 | 1.530 | 0.609 | 4.27 | 0 |
| pretrained_13kg_icode_K128_H32 | 24 | 22 | 1 | 1.177 | 3.980 | 28.52 | 0 |
| pretrained_13kg_icode_K16_H20 | 24 | 18 | 2 | 2.196 | 1.894 | 12.17 | 0 |
| pretrained_13kg_icode_K64_H24 | 24 | 22 | 1 | 1.202 | 2.398 | 17.48 | 0 |
| pretrained_13kg_nominal_K128_H32 | 24 | 23 | 0 | 0.754 | 0.811 | 5.45 | 0 |
| pretrained_13kg_nominal_K16_H20 | 24 | 19 | 2 | 2.029 | 0.553 | 3.73 | 0 |
| pretrained_13kg_nominal_K64_H24 | 24 | 23 | 1 | 1.035 | 0.572 | 4.29 | 0 |

## independent_validation_2026-09-09

| 方法 | n | 成功 | 碰撞 | 平均Q | 计算秒/回合 | 平均决策ms | deadline miss |
|---|---|---|---|---|---|---|---|
| causal_progress_feedback | 48 | 44 | 1 | 1.041 | 2.583 | 17.71 | 0 |
| fixed_icode_K256_H24 | 48 | 46 | 1 | 0.907 | 4.573 | 31.93 | 0 |
| fixed_icode_K64_H24 | 48 | 45 | 1 | 0.981 | 2.541 | 17.64 | 0 |
| fixed_nominal_K256_H24 | 48 | 45 | 1 | 0.978 | 0.768 | 5.37 | 0 |
| fixed_nominal_K64_H24 | 48 | 45 | 1 | 0.995 | 0.620 | 4.29 | 0 |
| ind_icode_H_full_s9091301_checkpoint_060066 | 48 | 46 | 2 | 1.044 | 3.369 | 24.70 | 0 |
| ind_icode_H_full_s9091302_checkpoint_060017 | 48 | 44 | 1 | 1.079 | 2.800 | 19.08 | 0 |
| ind_icode_KH_full_s9091301_checkpoint_060165 | 48 | 46 | 1 | 0.886 | 3.302 | 23.87 | 0 |
| ind_icode_KH_full_s9091302_checkpoint_060084 | 48 | 46 | 1 | 0.893 | 4.010 | 28.83 | 1 |
| ind_icode_KH_masked_s9091301_checkpoint_060156 | 48 | 45 | 1 | 0.977 | 3.583 | 25.37 | 0 |
| ind_icode_KH_masked_s9091302_checkpoint_060113 | 48 | 47 | 1 | 0.820 | 3.042 | 21.95 | 0 |
| ind_icode_K_full_s9091301_checkpoint_060076 | 48 | 46 | 1 | 0.899 | 4.522 | 32.32 | 0 |
| ind_icode_K_full_s9091302_checkpoint_060120 | 48 | 45 | 1 | 0.983 | 3.771 | 26.70 | 0 |

### 独立初始化：每个 seed 均保留

所有策略使用预先规定的训练终点检查点，不按评估表现挑选 seed 或检查点。两个初始化的训练环境部分共享；独立初始化不代表独立训练数据。以下均为联合 KH 减配对参照，同一个固定参照只运行一次，不重复算作独立样本。

初始化 9091301：`ind_icode_KH_full_s9091301_checkpoint_060165`

| 参照 | 成功率差pp | Q差 | Q诊断95%区间 | 计算秒差 |
|---|---|---|---|---|
| H | 0.0 | -0.158 | [-0.902, 0.276] | -0.067 |
| K | 0.0 | -0.014 | [-0.038, 0.013] | -1.219 |
| KH_masked | 2.1 | -0.092 | [-0.375, 0.021] | -0.281 |
| fixed_nominal_K256_H24 | 2.1 | -0.092 | [-0.628, 0.292] | 2.535 |
| fixed_nominal_K64_H24 | 2.1 | -0.109 | [-0.387, 0.004] | 2.683 |
| causal_progress_feedback | 4.2 | -0.155 | [-0.592, 0.005] | 0.719 |
| fixed_icode_K256_H24 | 0.0 | -0.021 | [-0.356, 0.298] | -1.270 |
| fixed_icode_K64_H24 | 2.1 | -0.095 | [-0.329, -0.001] | 0.761 |

初始化 9091302：`ind_icode_KH_full_s9091302_checkpoint_060084`

| 参照 | 成功率差pp | Q差 | Q诊断95%区间 | 计算秒差 |
|---|---|---|---|---|
| H | 4.2 | -0.186 | [-0.523, -0.007] | 1.210 |
| K | 2.1 | -0.090 | [-0.654, 0.319] | 0.239 |
| KH_masked | -2.1 | 0.074 | [-0.020, 0.314] | 0.968 |
| fixed_nominal_K256_H24 | 2.1 | -0.084 | [-0.622, 0.312] | 3.242 |
| fixed_nominal_K64_H24 | 2.1 | -0.101 | [-0.381, 0.005] | 3.390 |
| causal_progress_feedback | 4.2 | -0.147 | [-0.590, 0.029] | 1.427 |
| fixed_icode_K256_H24 | 0.0 | -0.013 | [-0.351, 0.317] | -0.563 |
| fixed_icode_K64_H24 | 2.1 | -0.087 | [-0.325, -0.000] | 1.469 |

这是两个初始化、两个环境 seed 的开发复验。区间描述各初始化的场景差异，不估计初始化总体不确定性，不据此宣称显著性或非劣性；需看全部参照及失败场景。

## mechanism_validation_2026-09-09

| 方法 | n | 成功 | 碰撞 | 平均Q | 计算秒/回合 | 平均决策ms | deadline miss |
|---|---|---|---|---|---|---|---|
| fixed_icode_K128_H27 | 48 | 46 | 0 | 0.766 | 3.529 | 23.96 | 0 |
| fixed_icode_K160_H29 | 48 | 46 | 0 | 0.776 | 4.246 | 28.37 | 0 |
| fixed_icode_K256_H24 | 48 | 46 | 1 | 0.913 | 4.425 | 30.26 | 0 |
| fixed_nominal_K128_H27 | 48 | 45 | 1 | 1.000 | 0.709 | 4.80 | 2 |
| fixed_nominal_K160_H29 | 48 | 45 | 1 | 0.998 | 0.765 | 5.12 | 0 |
| fixed_nominal_K256_H24 | 48 | 46 | 1 | 0.922 | 0.752 | 5.16 | 0 |
| joint_full_s9091301_icode | 48 | 46 | 0 | 0.750 | 3.318 | 22.95 | 0 |
| joint_full_s9091301_icode_hold_H | 48 | 44 | 2 | 1.207 | 2.829 | 19.88 | 1 |
| joint_full_s9091301_icode_hold_K | 48 | 47 | 0 | 0.687 | 3.651 | 25.13 | 0 |
| joint_full_s9091301_icode_hold_both | 48 | 44 | 2 | 1.219 | 3.080 | 21.43 | 0 |
| joint_full_s9091301_nominal | 48 | 45 | 1 | 0.972 | 0.668 | 4.70 | 0 |
| joint_full_s9091302_icode | 48 | 45 | 2 | 1.125 | 3.919 | 28.22 | 3 |
| joint_full_s9091302_icode_hold_H | 48 | 36 | 1 | 1.907 | 2.773 | 16.25 | 0 |
| joint_full_s9091302_icode_hold_K | 48 | 40 | 2 | 1.592 | 2.553 | 16.58 | 1 |
| joint_full_s9091302_icode_hold_both | 48 | 31 | 2 | 2.522 | 1.768 | 9.89 | 0 |
| joint_full_s9091302_nominal | 48 | 43 | 3 | 1.445 | 0.747 | 5.20 | 0 |
| joint_masked_s9091301_icode | 48 | 42 | 2 | 1.392 | 3.574 | 24.18 | 0 |
| joint_masked_s9091301_nominal | 48 | 44 | 2 | 1.213 | 0.691 | 4.84 | 0 |
| joint_masked_s9091302_icode | 48 | 46 | 1 | 0.912 | 3.110 | 21.62 | 0 |
| joint_masked_s9091302_nominal | 48 | 44 | 1 | 1.100 | 0.693 | 4.55 | 0 |

## model_screen

| 方法 | n | 成功 | 碰撞 | 平均Q | 计算秒/回合 | 平均决策ms | deadline miss |
|---|---|---|---|---|---|---|---|
| icode_K128_H20 | 24 | 16 | 2 | 2.665 | 3.563 | 20.51 | 1 |
| icode_K128_H32 | 24 | 17 | 2 | 2.438 | 5.207 | 30.31 | 0 |
| icode_K16_H12 | 24 | 9 | 1 | 3.862 | 1.916 | 8.46 | 0 |
| icode_K16_H20 | 24 | 14 | 1 | 2.781 | 2.550 | 12.77 | 0 |
| icode_K64_H20 | 24 | 15 | 2 | 2.856 | 2.935 | 16.18 | 0 |
| icode_K64_H32 | 24 | 16 | 1 | 2.344 | 4.213 | 23.47 | 0 |
| nominal_K128_H20 | 24 | 15 | 2 | 2.865 | 0.801 | 4.38 | 0 |
| nominal_K128_H32 | 24 | 14 | 2 | 3.028 | 1.150 | 6.09 | 2 |
| nominal_K16_H12 | 24 | 10 | 1 | 3.649 | 0.715 | 3.24 | 0 |
| nominal_K16_H20 | 24 | 14 | 2 | 3.008 | 0.697 | 3.77 | 0 |
| nominal_K64_H20 | 24 | 17 | 2 | 2.468 | 0.738 | 4.33 | 1 |
| nominal_K64_H32 | 24 | 16 | 1 | 2.373 | 0.965 | 5.26 | 0 |

## observable_validation

| 方法 | n | 成功 | 碰撞 | 平均Q | 计算秒/回合 | 平均决策ms | deadline miss |
|---|---|---|---|---|---|---|---|
| causal_progress_feedback | 48 | 44 | 2 | 1.202 | 2.388 | 16.65 | 0 |
| fixed_icode_K128_H32 | 48 | 46 | 2 | 1.058 | 3.882 | 27.45 | 0 |
| fixed_icode_K16_H20 | 48 | 43 | 1 | 1.163 | 1.741 | 11.56 | 0 |
| fixed_icode_K256_H24 | 48 | 46 | 1 | 0.934 | 4.456 | 30.05 | 0 |
| fixed_icode_K64_H24 | 48 | 45 | 2 | 1.131 | 2.410 | 16.82 | 0 |
| fixed_nominal_K128_H32 | 48 | 46 | 2 | 1.064 | 0.726 | 5.11 | 0 |
| fixed_nominal_K16_H20 | 48 | 44 | 0 | 0.938 | 0.531 | 3.49 | 0 |
| fixed_nominal_K256_H24 | 48 | 47 | 0 | 0.708 | 0.740 | 4.99 | 0 |
| fixed_nominal_K64_H24 | 48 | 45 | 2 | 1.131 | 0.578 | 4.05 | 0 |
| obs_icode_H_full_s9091201_checkpoint_015049 | 48 | 47 | 1 | 0.827 | 2.817 | 20.14 | 0 |
| obs_icode_H_full_s9091201_checkpoint_030128 | 48 | 46 | 1 | 0.898 | 2.767 | 19.75 | 0 |
| obs_icode_H_full_s9091201_checkpoint_045083 | 48 | 47 | 1 | 0.824 | 2.644 | 18.88 | 0 |
| obs_icode_H_full_s9091201_checkpoint_060224 | 48 | 46 | 1 | 0.899 | 3.513 | 25.12 | 0 |
| obs_icode_KH_full_s9091201_checkpoint_015064 | 48 | 46 | 1 | 0.908 | 3.269 | 23.02 | 0 |
| obs_icode_KH_full_s9091201_checkpoint_030123 | 48 | 46 | 1 | 0.904 | 3.785 | 26.44 | 0 |
| obs_icode_KH_full_s9091201_checkpoint_045007 | 48 | 47 | 0 | 0.678 | 3.970 | 27.41 | 0 |
| obs_icode_KH_full_s9091201_checkpoint_060096 | 48 | 48 | 0 | 0.607 | 4.119 | 28.68 | 0 |
| obs_icode_KH_masked_s9091201_checkpoint_015028 | 48 | 47 | 1 | 0.832 | 3.307 | 23.36 | 0 |
| obs_icode_KH_masked_s9091201_checkpoint_030099 | 48 | 48 | 0 | 0.609 | 3.492 | 24.25 | 0 |
| obs_icode_KH_masked_s9091201_checkpoint_045071 | 48 | 47 | 1 | 0.824 | 3.238 | 23.23 | 0 |
| obs_icode_KH_masked_s9091201_checkpoint_060045 | 48 | 45 | 0 | 0.805 | 3.396 | 23.24 | 0 |
| obs_icode_K_full_s9091201_checkpoint_015076 | 48 | 47 | 1 | 0.827 | 3.668 | 26.20 | 0 |
| obs_icode_K_full_s9091201_checkpoint_030027 | 48 | 47 | 1 | 0.833 | 3.817 | 26.96 | 0 |
| obs_icode_K_full_s9091201_checkpoint_045070 | 48 | 46 | 2 | 1.056 | 3.593 | 25.80 | 0 |
| obs_icode_K_full_s9091201_checkpoint_060228 | 48 | 47 | 1 | 0.830 | 3.997 | 28.31 | 0 |

按预先记录的成功优先、碰撞、Q规则选择的开发集固定参照：`fixed_icode_K256_H24`。

| 方法 | 成功率差pp | 保守95%区间 | Q差 | Q重采样95%区间 | 计算秒差 |
|---|---|---|---|---|---|
| obs_icode_KH_masked_s9091201_checkpoint_045071 | 2.1 | [-8.7, 12.6] | -0.110 | [-0.398, -0.003] | -1.218 |
| obs_icode_H_full_s9091201_checkpoint_060224 | 0.0 | [-12.5, 12.5] | -0.035 | [-0.372, 0.280] | -0.944 |
| obs_icode_KH_full_s9091201_checkpoint_045007 | 2.1 | [-12.2, 15.9] | -0.256 | [-1.130, 0.144] | -0.487 |
| obs_icode_KH_full_s9091201_checkpoint_060096 | 4.2 | [-8.4, 15.9] | -0.327 | [-1.256, 0.001] | -0.338 |
| obs_icode_K_full_s9091201_checkpoint_015076 | 2.1 | [-12.2, 15.9] | -0.107 | [-0.386, 0.002] | -0.789 |
| obs_icode_H_full_s9091201_checkpoint_030128 | 0.0 | [-12.5, 12.5] | -0.036 | [-0.397, 0.299] | -1.689 |
| obs_icode_KH_masked_s9091201_checkpoint_060045 | -2.1 | [-15.9, 12.2] | -0.129 | [-0.995, 0.513] | -1.060 |
| causal_progress_feedback | -4.2 | [-15.9, 8.4] | 0.268 | [-0.058, 1.144] | -2.068 |
| obs_icode_H_full_s9091201_checkpoint_045083 | 2.1 | [-12.2, 15.9] | -0.110 | [-1.260, 0.878] | -1.812 |
| obs_icode_H_full_s9091201_checkpoint_015049 | 2.1 | [-8.7, 12.6] | -0.107 | [-0.381, -0.003] | -1.639 |
| obs_icode_KH_full_s9091201_checkpoint_030123 | 0.0 | [-8.7, 8.7] | -0.030 | [-0.097, 0.018] | -0.671 |
| obs_icode_K_full_s9091201_checkpoint_045070 | 0.0 | [-12.5, 12.5] | 0.122 | [-0.351, 0.904] | -0.863 |
| obs_icode_KH_masked_s9091201_checkpoint_030099 | 4.2 | [-8.4, 15.9] | -0.325 | [-1.244, 0.001] | -0.964 |
| obs_icode_K_full_s9091201_checkpoint_060228 | 2.1 | [-8.7, 12.6] | -0.104 | [-0.374, -0.004] | -0.459 |
| obs_icode_KH_masked_s9091201_checkpoint_015028 | 2.1 | [-8.7, 12.6] | -0.102 | [-0.384, 0.003] | -1.149 |
| obs_icode_K_full_s9091201_checkpoint_030027 | 2.1 | [-12.2, 15.9] | -0.101 | [-0.380, 0.005] | -0.639 |
| obs_icode_KH_full_s9091201_checkpoint_015064 | 0.0 | [-8.7, 8.7] | -0.026 | [-0.064, 0.002] | -1.187 |

成功率区间基于配对不一致事件的保守区间，仍有回合独立性假设；Q区间按场景和seed双向配对重采样。开发集用于选模型，这些区间只用于诊断，不是选模后无偏的最终推断。少量seed也限制了不确定性估计。

### 按环境 seed 交叉验证的场景预算空间

| 预测模型 | 固定参照Q | 场景oracle Q | Q相对空间 | 成功率差pp | 计算秒差 |
|---|---|---|---|---|---|
| icode | 1.069 | 0.834 | 22.0% | 2.1 | -1.205 |
| nominal | 1.136 | 1.060 | 6.7% | 2.1 | -0.036 |

每一折只用其他环境seed选择全局固定预算和各场景/速度预算，再在留出的seed计分。该oracle知道场景标签，只覆盖已测固定候选，不包含回合内切换或选择开销。少量seed、折间依赖和已有开发选择限制了推断；这里只诊断剩余空间，不作最终Go判据。

### 联合调度的直接配对比较

联合组检查点：`obs_icode_KH_full_s9091201_checkpoint_060096`。各训练组按成功、碰撞、Q、计算量顺序选择开发检查点。下表均为联合组减参照组；所有检查点的完整结果保留在前表。

| 参照组 | 参照检查点/方法 | 成功率差pp | Q差 | Q重采样95%区间 | 计算秒差 |
|---|---|---|---|---|---|
| KH_masked | obs_icode_KH_masked_s9091201_checkpoint_030099 | 0.0 | -0.002 | [-0.021, 0.015] | 0.627 |
| H | obs_icode_H_full_s9091201_checkpoint_045083 | 2.1 | -0.217 | [-0.882, 0.016] | 1.475 |
| K | obs_icode_K_full_s9091201_checkpoint_015076 | 2.1 | -0.220 | [-0.900, 0.020] | 0.451 |
| fixed | fixed_icode_K256_H24 | 4.2 | -0.327 | [-1.256, 0.001] | -0.338 |
| heuristic | causal_progress_feedback | 8.3 | -0.595 | [-1.489, 0.000] | 1.731 |

这是开发集上选模后的诊断，区间没有修正检查点选择偏差，不能据此单独宣称正式优势。

## shared_backbone_budget_check

| 方法 | n | 成功 | 碰撞 | 平均Q | 计算秒/回合 | 平均决策ms | deadline miss |
|---|---|---|---|---|---|---|---|
| t2.5_base_i0.7_icode_K128_H32 | 24 | 15 | 2 | 2.799 | 5.241 | 28.77 | 0 |
| t2.5_base_i0.7_icode_K16_H20 | 24 | 13 | 1 | 3.003 | 2.499 | 12.11 | 0 |
| t2.5_base_i0.7_icode_K64_H24 | 24 | 14 | 2 | 3.001 | 3.159 | 17.25 | 0 |
| t2.5_base_i0.7_nominal_K128_H32 | 24 | 14 | 1 | 2.754 | 1.054 | 5.32 | 0 |
| t2.5_base_i0.7_nominal_K16_H20 | 24 | 12 | 1 | 3.146 | 0.760 | 3.66 | 0 |
| t2.5_base_i0.7_nominal_K64_H24 | 24 | 14 | 2 | 3.000 | 0.794 | 4.29 | 0 |
| t2.5_wide_i0.45_icode_K128_H32 | 24 | 22 | 2 | 1.525 | 4.045 | 28.43 | 0 |
| t2.5_wide_i0.45_icode_K16_H20 | 24 | 15 | 2 | 2.815 | 2.277 | 12.22 | 0 |
| t2.5_wide_i0.45_icode_K64_H24 | 24 | 19 | 1 | 1.777 | 2.757 | 17.36 | 0 |
| t2.5_wide_i0.45_nominal_K128_H32 | 24 | 21 | 2 | 1.678 | 0.784 | 5.30 | 0 |
| t2.5_wide_i0.45_nominal_K16_H20 | 24 | 17 | 2 | 2.435 | 0.622 | 3.72 | 0 |
| t2.5_wide_i0.45_nominal_K64_H24 | 24 | 18 | 1 | 1.935 | 0.714 | 4.24 | 0 |
| t2.5_wide_i0.7_icode_K128_H32 | 24 | 23 | 1 | 1.097 | 4.309 | 28.38 | 0 |
| t2.5_wide_i0.7_icode_K16_H20 | 24 | 20 | 1 | 1.635 | 1.910 | 12.02 | 0 |
| t2.5_wide_i0.7_icode_K64_H24 | 24 | 21 | 0 | 1.149 | 2.954 | 17.68 | 0 |
| t2.5_wide_i0.7_nominal_K128_H32 | 24 | 23 | 1 | 1.094 | 0.816 | 5.39 | 0 |
| t2.5_wide_i0.7_nominal_K16_H20 | 24 | 20 | 1 | 1.626 | 0.588 | 3.73 | 0 |
| t2.5_wide_i0.7_nominal_K64_H24 | 24 | 23 | 0 | 0.815 | 0.687 | 4.28 | 0 |

## shared_backbone_factorial

| 方法 | n | 成功 | 碰撞 | 平均Q | 计算秒/回合 | 平均决策ms | deadline miss |
|---|---|---|---|---|---|---|---|
| t10_base_i0.45_icode_K64_H24 | 24 | 13 | 2 | 3.247 | 3.791 | 18.71 | 0 |
| t10_base_i0.7_icode_K64_H24 | 24 | 12 | 2 | 3.391 | 3.709 | 18.59 | 0 |
| t10_wide_i0.45_icode_K64_H24 | 24 | 18 | 2 | 2.257 | 3.117 | 18.52 | 0 |
| t10_wide_i0.7_icode_K64_H24 | 24 | 20 | 2 | 1.943 | 2.998 | 18.06 | 0 |
| t2.5_base_i0.45_icode_K64_H24 | 24 | 13 | 2 | 3.227 | 3.578 | 18.38 | 0 |
| t2.5_base_i0.7_icode_K64_H24 | 24 | 13 | 2 | 3.187 | 3.624 | 18.46 | 0 |
| t2.5_wide_i0.45_icode_K64_H24 | 24 | 21 | 1 | 1.449 | 3.046 | 19.29 | 4 |
| t2.5_wide_i0.7_icode_K64_H24 | 24 | 20 | 2 | 1.882 | 2.854 | 18.19 | 0 |

## validation_4

| 方法 | n | 成功 | 碰撞 | 平均Q | 计算秒/回合 | 平均决策ms | deadline miss |
|---|---|---|---|---|---|---|---|
| fixed_icode_K128_H32 | 48 | 36 | 2 | 1.966 | 5.179 | 29.89 | 0 |
| fixed_icode_K16_H20 | 48 | 25 | 4 | 3.322 | 2.507 | 12.83 | 0 |
| fixed_icode_K256_H24 | 48 | 35 | 3 | 2.235 | 5.688 | 32.70 | 0 |
| fixed_icode_K64_H24 | 48 | 32 | 3 | 2.480 | 3.278 | 18.65 | 0 |
| fixed_nominal_K128_H32 | 48 | 30 | 4 | 2.819 | 1.086 | 5.73 | 0 |
| fixed_nominal_K16_H20 | 48 | 25 | 4 | 3.315 | 0.772 | 3.95 | 0 |
| fixed_nominal_K256_H24 | 48 | 35 | 3 | 2.229 | 0.966 | 5.59 | 0 |
| fixed_nominal_K64_H24 | 48 | 33 | 3 | 2.396 | 0.773 | 4.44 | 0 |
| sac_icode_H_full_s9091001_checkpoint_010009 | 48 | 30 | 3 | 2.668 | 6.486 | 35.64 | 0 |
| sac_icode_H_full_s9091001_checkpoint_020208 | 48 | 25 | 3 | 3.222 | 3.887 | 19.49 | 0 |
| sac_icode_H_full_s9091001_checkpoint_030047 | 48 | 27 | 2 | 2.905 | 3.846 | 19.29 | 0 |
| sac_icode_H_full_s9091001_checkpoint_040227 | 48 | 32 | 3 | 2.549 | 4.593 | 25.02 | 0 |
| sac_icode_KH_full_s9091001_checkpoint_010296 | 48 | 20 | 2 | 3.645 | 3.191 | 14.10 | 0 |
| sac_icode_KH_full_s9091001_checkpoint_020199 | 48 | 20 | 3 | 3.723 | 3.387 | 15.50 | 0 |
| sac_icode_KH_full_s9091001_checkpoint_030193 | 48 | 30 | 4 | 2.861 | 3.752 | 20.76 | 0 |
| sac_icode_KH_full_s9091001_checkpoint_040017 | 48 | 23 | 3 | 3.427 | 3.515 | 16.85 | 0 |
| sac_icode_KH_masked_s9091001_checkpoint_010079 | 48 | 32 | 3 | 2.529 | 3.836 | 21.25 | 1 |
| sac_icode_KH_masked_s9091001_checkpoint_020022 | 48 | 26 | 3 | 3.073 | 3.440 | 17.59 | 0 |
| sac_icode_KH_masked_s9091001_checkpoint_030022 | 48 | 27 | 3 | 3.009 | 3.377 | 17.63 | 0 |
| sac_icode_KH_masked_s9091001_checkpoint_040022 | 48 | 28 | 4 | 3.018 | 4.014 | 21.36 | 0 |
| sac_icode_K_full_s9091001_checkpoint_010098 | 48 | 33 | 3 | 2.361 | 5.476 | 31.80 | 0 |
| sac_icode_K_full_s9091001_checkpoint_020021 | 48 | 32 | 3 | 2.498 | 5.656 | 31.21 | 0 |
| sac_icode_K_full_s9091001_checkpoint_030163 | 48 | 34 | 4 | 2.450 | 5.202 | 30.24 | 0 |
| sac_icode_K_full_s9091001_checkpoint_040127 | 48 | 34 | 3 | 2.307 | 5.886 | 33.92 | 0 |

按预先记录的成功优先、碰撞、Q规则选择的开发集固定参照：`fixed_icode_K128_H32`。

| 方法 | 成功率差pp | 保守95%区间 | Q差 | Q重采样95%区间 | 计算秒差 |
|---|---|---|---|---|---|
| sac_icode_KH_masked_s9091001_checkpoint_040022 | -16.7 | [-32.2, 2.1] | 1.052 | [0.239, 2.112] | -1.164 |
| sac_icode_K_full_s9091001_checkpoint_030163 | -4.2 | [-18.9, 11.6] | 0.484 | [-0.148, 1.473] | 0.023 |
| sac_icode_KH_masked_s9091001_checkpoint_010079 | -8.3 | [-26.8, 11.8] | 0.562 | [-0.194, 1.592] | -1.342 |
| sac_icode_KH_masked_s9091001_checkpoint_030022 | -18.8 | [-37.0, 3.2] | 1.043 | [0.296, 1.843] | -1.801 |
| sac_icode_KH_masked_s9091001_checkpoint_020022 | -20.8 | [-37.0, -0.7] | 1.107 | [0.408, 1.918] | -1.739 |
| sac_icode_H_full_s9091001_checkpoint_010009 | -12.5 | [-27.1, 4.7] | 0.702 | [0.018, 1.909] | 1.308 |
| sac_icode_KH_full_s9091001_checkpoint_020199 | -33.3 | [-50.4, -10.2] | 1.757 | [0.851, 2.752] | -1.792 |
| sac_icode_H_full_s9091001_checkpoint_030047 | -18.8 | [-34.6, 0.8] | 0.939 | [0.262, 1.661] | -1.333 |
| sac_icode_H_full_s9091001_checkpoint_020208 | -22.9 | [-39.3, -2.2] | 1.256 | [0.419, 2.359] | -1.291 |
| sac_icode_K_full_s9091001_checkpoint_040127 | -4.2 | [-18.9, 11.6] | 0.341 | [-0.355, 1.325] | 0.708 |
| sac_icode_KH_full_s9091001_checkpoint_010296 | -33.3 | [-50.4, -10.2] | 1.679 | [0.695, 2.785] | -1.987 |
| sac_icode_KH_full_s9091001_checkpoint_040017 | -27.1 | [-43.9, -5.2] | 1.461 | [0.457, 2.665] | -1.664 |
| sac_icode_KH_full_s9091001_checkpoint_030193 | -12.5 | [-29.7, 7.3] | 0.895 | [0.044, 1.993] | -1.427 |
| sac_icode_K_full_s9091001_checkpoint_010098 | -6.2 | [-21.7, 10.7] | 0.395 | [-0.371, 1.429] | 0.297 |
| sac_icode_H_full_s9091001_checkpoint_040227 | -8.3 | [-24.5, 9.7] | 0.583 | [0.031, 1.422] | -0.586 |
| sac_icode_K_full_s9091001_checkpoint_020021 | -8.3 | [-21.8, 6.8] | 0.532 | [0.009, 1.325] | 0.477 |

成功率区间基于配对不一致事件的保守区间，仍有回合独立性假设；Q区间按场景和seed双向配对重采样。开发集用于选模型，这些区间只用于诊断，不是选模后无偏的最终推断。少量seed也限制了不确定性估计。

### 按环境 seed 交叉验证的场景预算空间

| 预测模型 | 固定参照Q | 场景oracle Q | Q相对空间 | 成功率差pp | 计算秒差 |
|---|---|---|---|---|---|
| icode | 2.314 | 2.216 | 4.3% | 2.1 | -1.178 |
| nominal | 2.229 | 2.458 | -10.3% | -2.1 | -0.092 |

每一折只用其他环境seed选择全局固定预算和各场景/速度预算，再在留出的seed计分。该oracle知道场景标签，只覆盖已测固定候选，不包含回合内切换或选择开销。少量seed、折间依赖和已有开发选择限制了推断；这里只诊断剩余空间，不作最终Go判据。

### 联合调度的直接配对比较

联合组检查点：`sac_icode_KH_full_s9091001_checkpoint_030193`。各训练组按成功、碰撞、Q、计算量顺序选择开发检查点。下表均为联合组减参照组；所有检查点的完整结果保留在前表。

| 参照组 | 参照检查点/方法 | 成功率差pp | Q差 | Q重采样95%区间 | 计算秒差 |
|---|---|---|---|---|---|
| KH_masked | sac_icode_KH_masked_s9091001_checkpoint_010079 | -4.2 | 0.332 | [-0.176, 1.129] | -0.085 |
| K | sac_icode_K_full_s9091001_checkpoint_040127 | -8.3 | 0.554 | [-0.032, 1.341] | -2.135 |
| H | sac_icode_H_full_s9091001_checkpoint_040227 | -4.2 | 0.312 | [-0.390, 0.972] | -0.841 |
| fixed | fixed_icode_K128_H32 | -12.5 | 0.895 | [0.044, 1.993] | -1.427 |

这是开发集上选模后的诊断，区间没有修正检查点选择偏差，不能据此单独宣称正式优势。

## 相同指令历史上的两域预测诊断

| 物理配置 | 产生轨迹的模型 | 速度 | H | 回合 | ICODE位置误差差值m | ICODE归一化误差差值 |
|---|---|---|---|---|---|---|
| current_9kg | nominal | moderate | 1 | 12 | 0.0000 | -0.0072 |
| current_9kg | nominal | moderate | 8 | 12 | -0.0013 | -0.0263 |
| current_9kg | nominal | moderate | 20 | 12 | -0.0160 | -0.0635 |
| current_9kg | nominal | moderate | 32 | 12 | -0.0371 | -0.1011 |
| current_9kg | nominal | fast | 1 | 12 | -0.0002 | -0.0116 |
| current_9kg | nominal | fast | 8 | 12 | -0.0141 | -0.0754 |
| current_9kg | nominal | fast | 20 | 12 | -0.0598 | -0.1845 |
| current_9kg | nominal | fast | 32 | 12 | -0.1259 | -0.3251 |
| current_9kg | icode | moderate | 1 | 12 | 0.0000 | -0.0102 |
| current_9kg | icode | moderate | 8 | 12 | -0.0018 | -0.0330 |
| current_9kg | icode | moderate | 20 | 12 | -0.0166 | -0.0662 |
| current_9kg | icode | moderate | 32 | 12 | -0.0365 | -0.1029 |
| current_9kg | icode | fast | 1 | 12 | -0.0001 | -0.0167 |
| current_9kg | icode | fast | 8 | 12 | -0.0141 | -0.0739 |
| current_9kg | icode | fast | 20 | 12 | -0.0624 | -0.2039 |
| current_9kg | icode | fast | 32 | 12 | -0.1331 | -0.3453 |
| pretrained_13kg | nominal | moderate | 1 | 12 | 0.0000 | -0.0058 |
| pretrained_13kg | nominal | moderate | 8 | 12 | 0.0004 | -0.0175 |
| pretrained_13kg | nominal | moderate | 20 | 12 | -0.0055 | -0.0364 |
| pretrained_13kg | nominal | moderate | 32 | 12 | -0.0175 | -0.0618 |
| pretrained_13kg | nominal | fast | 1 | 12 | -0.0001 | -0.0117 |
| pretrained_13kg | nominal | fast | 8 | 12 | -0.0119 | -0.0593 |
| pretrained_13kg | nominal | fast | 20 | 12 | -0.0559 | -0.1530 |
| pretrained_13kg | nominal | fast | 32 | 12 | -0.1125 | -0.2541 |
| pretrained_13kg | icode | moderate | 1 | 12 | 0.0000 | -0.0076 |
| pretrained_13kg | icode | moderate | 8 | 12 | 0.0004 | -0.0242 |
| pretrained_13kg | icode | moderate | 20 | 12 | -0.0061 | -0.0424 |
| pretrained_13kg | icode | moderate | 32 | 12 | -0.0185 | -0.0631 |
| pretrained_13kg | icode | fast | 1 | 12 | -0.0001 | -0.0144 |
| pretrained_13kg | icode | fast | 8 | 12 | -0.0122 | -0.0707 |
| pretrained_13kg | icode | fast | 20 | 12 | -0.0602 | -0.1683 |
| pretrained_13kg | icode | fast | 32 | 12 | -0.1230 | -0.2846 |

差值为ICODE减nominal，负值表示ICODE误差更低。两模型使用同一批已记录的后续执行指令；先在每回合内计算RMSE，再对回合等权平均。该离线条件预测不等同于规划反事实预测。全部horizon共用完整32周期窗口，短回合排除0个，保留排除名单；每组只有一个环境seed。

## 因果输入与物理条件

第一轮使用9kg、扭矩上限2.2的物理配置，而冻结L57 ICODE来自13kg、扭矩上限1.55、较低摩擦配置。两者均为40ms执行延迟。后续两域对照和13kg训练由预训练来源决定，所有方法使用相同物理配置。

第一轮innovation从仿真器记录的执行指令均值计算。已准备从发送指令、已测readiness与固定40ms校准延迟重建的核对；后续observable接口只使用该指令历史估计和感知状态，屏蔽实际内部执行均值。support属于归一化范围指标，不能解释为校准后的可靠性概率。

SAC确实使用tanh Gaussian actor、双Q、target Q及熵调节；replay保存未取整动作，环境取整为K/H。每个选择保持5周期，策略推断与上下文均计入readiness。训练器优化在仿真步之间离线进行，未算作部署推断时间。

Q = duration/30 + 2×final_distance/4.5 + 3×timeout + 10×collision；计算价格不在Q中。第一轮训练价格0.20，后续0.05。它们是显式计算偏好，不是自然物理退化证据。

第一轮验证在1152回合完成后出现长时间退出清理，进程约5.9GB常驻内存、1.1GB交换内存。该资源积累可能影响后半批延迟；配对组内随机顺序不能完全消除这一限制。后续增加逐回合资源日志，大批验证按完整配对场景使用独立串行进程，减少内存积累造成的计时偏差。

## 原始轨迹算术及来源核对

| 阶段 | 回合 | 周期 | 通过 |
|---|---|---|---|
| icode_domain_screen | 288 | 42583 | True |
| ind_icode_H_full_s9091301 | 420 | 60066 | True |
| ind_icode_H_full_s9091302 | 417 | 60017 | True |
| ind_icode_KH_full_s9091301 | 411 | 60165 | True |
| ind_icode_KH_full_s9091302 | 427 | 60084 | True |
| ind_icode_KH_masked_s9091301 | 413 | 60156 | True |
| ind_icode_KH_masked_s9091302 | 418 | 60113 | True |
| ind_icode_K_full_s9091301 | 423 | 60076 | True |
| ind_icode_K_full_s9091302 | 429 | 60120 | True |
| independent_validation_2026-09-09 | 624 | 88413 | True |
| mechanism_validation_2026-09-09 | 960 | 142880 | True |
| model_screen | 288 | 54323 | True |
| obs_icode_H_full_s9091201 | 423 | 60224 | True |
| obs_icode_KH_full_s9091201 | 414 | 60096 | True |
| obs_icode_KH_masked_s9091201 | 411 | 60045 | True |
| obs_icode_K_full_s9091201 | 423 | 60228 | True |
| observable_validation | 1200 | 171789 | True |
| sac_icode_H_full_s9091001 | 215 | 40227 | True |
| sac_icode_KH_full_s9091001 | 200 | 40017 | True |
| sac_icode_KH_masked_s9091001 | 216 | 40022 | True |
| sac_icode_K_full_s9091001 | 227 | 40127 | True |
| shared_backbone_budget_check | 432 | 73943 | True |
| shared_backbone_factorial | 192 | 34609 | True |
| validation_4 | 1152 | 216180 | True |

## 指令历史可重建性

| 阶段 | 回合 | 周期 | 通过 |
|---|---|---|---|
| icode_domain_screen | 288 | 42583 | True |
| ind_icode_H_full_s9091301 | 420 | 60066 | True |
| ind_icode_H_full_s9091302 | 417 | 60017 | True |
| ind_icode_KH_full_s9091301 | 411 | 60165 | True |
| ind_icode_KH_full_s9091302 | 427 | 60084 | True |
| ind_icode_KH_masked_s9091301 | 413 | 60156 | True |
| ind_icode_KH_masked_s9091302 | 418 | 60113 | True |
| ind_icode_K_full_s9091301 | 423 | 60076 | True |
| ind_icode_K_full_s9091302 | 429 | 60120 | True |
| independent_validation_2026-09-09 | 624 | 88413 | True |
| mechanism_validation_2026-09-09 | 960 | 142880 | True |
| model_screen | 288 | 54323 | True |
| obs_icode_H_full_s9091201 | 423 | 60224 | True |
| obs_icode_KH_full_s9091201 | 414 | 60096 | True |
| obs_icode_KH_masked_s9091201 | 411 | 60045 | True |
| obs_icode_K_full_s9091201 | 423 | 60228 | True |
| observable_validation | 1200 | 171789 | True |
| sac_icode_H_full_s9091001 | 215 | 40227 | True |
| sac_icode_KH_full_s9091001 | 200 | 40017 | True |
| sac_icode_KH_masked_s9091001 | 216 | 40022 | True |
| sac_icode_K_full_s9091001 | 227 | 40127 | True |
| shared_backbone_budget_check | 432 | 73943 | True |
| shared_backbone_factorial | 192 | 34609 | True |
| validation_4 | 1152 | 216180 | True |

## 当前结论范围

本报告不把训练完成、单次成功率提高或计算节省当作ICRA级别结果。联合K/H优势、可靠性信息增益和ICODE模型增益需要分别通过匹配对照验证。全部9090301–9090304最终seed保留；当前所有选择均属开发。若开发候选有效，还需独立初始化重复、最终配对测试和相应置信区间。

证据根目录：`research_artifacts/icode_sac_compute_2026-09-08/`。每个阶段保留resolved configs、源码快照、原始轨迹、训练replay与检查点；队列错误和恢复也保留。协议位于`docs/protocols/icode_sac_*.md`。
