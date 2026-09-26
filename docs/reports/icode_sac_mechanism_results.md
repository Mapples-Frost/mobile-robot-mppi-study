# K/H 与 ICODE 机制实验
Exploratory prospective interventions; all contrasts reported. No multiplicity-adjusted significance, no noninferiority or best-seed selection. Nominal actors are transferred without retraining; fixed holding retains actor overhead.
| 方法 | 回合 | 成功 | 碰撞 | Q | 计算秒/回合 |
|---|---:|---:|---:|---:|---:|
| fixed_icode_K128_H27 | 48 | 46 | 0 | 0.766 | 3.529 |
| fixed_icode_K160_H29 | 48 | 46 | 0 | 0.776 | 4.246 |
| fixed_icode_K256_H24 | 48 | 46 | 1 | 0.913 | 4.425 |
| fixed_nominal_K128_H27 | 48 | 45 | 1 | 1.000 | 0.709 |
| fixed_nominal_K160_H29 | 48 | 45 | 1 | 0.998 | 0.765 |
| fixed_nominal_K256_H24 | 48 | 46 | 1 | 0.922 | 0.752 |
| joint_full_s9091301_icode | 48 | 46 | 0 | 0.750 | 3.318 |
| joint_full_s9091301_icode_hold_H | 48 | 44 | 2 | 1.207 | 2.829 |
| joint_full_s9091301_icode_hold_K | 48 | 47 | 0 | 0.687 | 3.651 |
| joint_full_s9091301_icode_hold_both | 48 | 44 | 2 | 1.219 | 3.080 |
| joint_full_s9091301_nominal | 48 | 45 | 1 | 0.972 | 0.668 |
| joint_full_s9091302_icode | 48 | 45 | 2 | 1.125 | 3.919 |
| joint_full_s9091302_icode_hold_H | 48 | 36 | 1 | 1.907 | 2.773 |
| joint_full_s9091302_icode_hold_K | 48 | 40 | 2 | 1.592 | 2.553 |
| joint_full_s9091302_icode_hold_both | 48 | 31 | 2 | 2.522 | 1.768 |
| joint_full_s9091302_nominal | 48 | 43 | 3 | 1.445 | 0.747 |
| joint_masked_s9091301_icode | 48 | 42 | 2 | 1.392 | 3.574 |
| joint_masked_s9091301_nominal | 48 | 44 | 2 | 1.213 | 0.691 |
| joint_masked_s9091302_icode | 48 | 46 | 1 | 0.912 | 3.110 |
| joint_masked_s9091302_nominal | 48 | 44 | 1 | 1.100 | 0.693 |

差值均为名称左侧减右侧；成功率差越高越好，Q和计算差越低越好。
| 配对比较 | 成功率差pp | Q差 | Q诊断95%区间 | 计算秒差 |
|---|---:|---:|---|---:|
| fixed_icode_K128_H27 minus fixed_nominal_K128_H27 | 2.08 | -0.233 | [-0.916, -0.003] | 2.820 |
| fixed_icode_K160_H29 minus fixed_nominal_K160_H29 | 2.08 | -0.223 | [-1.211, 0.342] | 3.481 |
| fixed_icode_K256_H24 minus fixed_nominal_K256_H24 | 0.00 | -0.009 | [-0.038, 0.006] | 3.673 |
| joint_full_s9091301_icode minus fixed_icode_K128_H27 | 0.00 | -0.016 | [-0.316, 0.284] | -0.211 |
| joint_full_s9091301_icode minus fixed_icode_K160_H29 | 0.00 | -0.026 | [-0.373, 0.298] | -0.928 |
| joint_full_s9091301_icode minus fixed_icode_K256_H24 | 0.00 | -0.163 | [-1.101, 0.379] | -1.107 |
| joint_full_s9091301_icode minus joint_full_s9091301_icode_hold_H | 4.17 | -0.457 | [-1.493, 0.292] | 0.489 |
| joint_full_s9091301_icode minus joint_full_s9091301_icode_hold_K | -2.08 | 0.063 | [-0.035, 0.306] | -0.333 |
| joint_full_s9091301_icode minus joint_full_s9091301_icode_hold_both | 4.17 | -0.468 | [-1.431, 0.153] | 0.238 |
| joint_full_s9091301_icode minus joint_full_s9091301_nominal | 2.08 | -0.222 | [-0.893, 0.010] | 2.651 |
| joint_full_s9091301_icode minus joint_masked_s9091301_icode | 8.33 | -0.642 | [-2.433, 0.005] | -0.256 |
| joint_full_s9091302_icode minus fixed_icode_K128_H27 | -2.08 | 0.359 | [-0.297, 1.336] | 0.390 |
| joint_full_s9091302_icode minus fixed_icode_K160_H29 | -2.08 | 0.349 | [-0.347, 1.340] | -0.327 |
| joint_full_s9091302_icode minus fixed_icode_K256_H24 | -2.08 | 0.212 | [-0.314, 1.196] | -0.506 |
| joint_full_s9091302_icode minus joint_full_s9091302_icode_hold_H | 18.75 | -0.781 | [-1.608, 0.290] | 1.146 |
| joint_full_s9091302_icode minus joint_full_s9091302_icode_hold_K | 10.42 | -0.466 | [-0.970, -0.009] | 1.366 |
| joint_full_s9091302_icode minus joint_full_s9091302_icode_hold_both | 29.17 | -1.397 | [-2.427, -0.477] | 2.151 |
| joint_full_s9091302_icode minus joint_full_s9091302_nominal | 4.17 | -0.319 | [-0.932, -0.007] | 3.172 |
| joint_full_s9091302_icode minus joint_masked_s9091302_icode | -2.08 | 0.213 | [-0.036, 0.899] | 0.809 |
| joint_masked_s9091301_icode minus joint_masked_s9091301_nominal | -4.17 | 0.179 | [-0.906, 1.254] | 2.882 |
| joint_masked_s9091302_icode minus joint_masked_s9091302_nominal | 4.17 | -0.188 | [-0.753, 0.007] | 2.417 |

这些结果仅针对当前仿真场景、理想传感、两个环境seed和两个初始化。切换预测模型没有重训actor，不能替代公平训练的nominal-SAC。锁定坐标仍支付actor开销，用于干预诊断；固定控制器是另行实测的部署比较。
