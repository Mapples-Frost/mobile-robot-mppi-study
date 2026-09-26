# AWS短训练可行性实测（2026-09-26）

范围：工程探针，不是正式复现结论；没有读取validation/test。t3a.medium，EPYC7571，2vCPU/4GiB。固定H25(vehicle)/H30(pendulum)，seed2609268801，各200步，前100步学习预热，batch64，buffer2000。与正式50000容量回放缓存不同，不能把本次内存当作完整训练上限。

vehicle：38.8967秒，101次梯度更新，200步均有限，参数实际变化且全部有限，0个求解失败步，两次goal终止。
pendulum：21.7766秒，101次梯度更新，200步均有限，参数实际变化且全部有限，9/200求解失败步；两个已结束回合分别steps与constraint。失败全部保留，不能把工程跑通当作控制成功。

控制step（含环境工作，非纯solver计时）：vehicle mean/median/P95为176.2/169.3/215.6ms；pendulum为93.4/80.5/116.7ms。整个连续进程峰值RSS427664KiB，约417.6MiB。两任务训练时间均不含初始化（各约5秒）。

仅按这次200步平均速度线性折算15000步：vehicle约48.6分钟，pendulum约27.2分钟。不是正式训练工期：完整训练更新比例更高、后期求解难度/策略会变、未含完整评估，也未验证CPU积分耗尽后的速度或费用。不能据此承诺全天持续吞吐。

探针运行期间bohn-research让出资源，由独立systemd单元运行，ExecStopPost恢复主服务；探针退出0，主服务已active。运行命令为legacy Python调用experiments/bohn2021_aws/training_feasibility_probe.py；脚本、配置、checkpoint、逐步记录与registry均保留。
