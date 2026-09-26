# 当前t3a.medium监控与最小只读权限

用户规则已写入scripts/research_service/RESOURCE_POLICY.md：继续使用当前实例；只有持续CPU credit成本、显著降速、队列成为主要瓶颈、或多seed/config并行确有明显工期收益时才提出升级建议，不能仅看2vCPU/4GiB。没有授权自动升级或修改IAM/计费模式。

已部署：每实验2秒本机/进程CPU采样、完整命令wall-clock、CPU秒数、峰值RSS；独立systemd timer每分钟记录主机CPU，每5分钟尝试获取AWS/EC2的CPUUtilization及三项credit指标。CPUSurplusCreditsCharged使用Sum，其单位是vCPU分钟，不是美元。五分钟边界、延迟上报和缺失值单列，不能用0替代不可用。

当前服务器无实例角色，也没有AWS凭据；因此credit数据目前不可用。采集器准备完成，配置只读角色后会自动补采。没有把缺失权限当成零费用，也没有宣称已记录完整credit数据。

最小权限策略见cloudwatch-minimal-read-policy.json，仅允许us-west-2的cloudwatch:GetMetricData。此处读取经典EC2指标不能使用EC2实例ARN来约束Resource；Resource:*允许该区域的指标读取，不授予修改权限。采集器固定查询实例i-0d05cbb43d4a1f54d及四个指标。

由用户/管理员在IAM控制台创建EC2用途角色（信任ec2.amazonaws.com），附加上面的自定义只读策略，再在俄勒冈EC2控制台选择实例→操作→安全→修改IAM角色，选择该角色。无需发送Access Key，不需要AdministratorAccess或CloudWatchFullAccess，不需要开启详细监控，也不需要重启/升级实例。本次只提供策略与说明，未执行IAM操作。

路径：/data/openai-agent/state/telemetry/host.jsonl（本机时间序列）；cloudwatch_status.json（权限/延迟/可用状态）；cloudwatch.json（按时间戳去重的CloudWatch序列）；每次实验目录cpu_samples.jsonl、registry.json、cloudwatch_snapshot.json。即时snapshot可能尚无最新CloudWatch点，最终报告须用持续补采后的全局序列关联训练时段。

采样公式、PID复用、缺失credit非零、charged Sum、实际采样5项测试通过；控制器8项检查通过；额外4秒CPU工作负载验证了wall-clock/CPU样本和缺失权限状态。

AWS参考：https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-performance-instances-monitoring-cpu-credits.html
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/attach-iam-role.html
