# Current-instance execution and upgrade policy (user instruction, 2026-09-26)

Continue the Bohn vehicle and inverted-pendulum research on the existing t3a.medium. No instance upgrade is authorized. Nominal 2vCPU/4GiB is not evidence of insufficiency. Complete formally preregistered training and retain all seed/config results, rather than deciding from smoke tests.

For every complete training command, retain UTC start/end, monotonic wall-clock duration, CPU seconds, two-second host/process CPU samples, peak RSS and raw logs. Separate interpreter/model construction, optimizer/rollout times and evaluation when available. Collector remains the same for comparison arms; disclose its small observation overhead. Whole-process wall time is observed at process reaping (up to two seconds polling delay). Record simulation steps/episodes/updates and throughput through the actual training runner.

Record CloudWatch AWS/EC2 CPUUtilization and CPUCreditBalance, CPUSurplusCreditBalance, CPUSurplusCreditsCharged, with instance ID, region, UTC timestamps, period and statistic. Credits are vCPU-minutes, NOT dollars. Credit metrics have five-minute granularity. Balances are gauges; charged credits use Sum per bucket. Overlapping retrievals are deduplicated by metric and timestamp. Missing credentials, no datapoints, delay and a true numeric zero are distinct. Do not claim zero cost from missing data. Actual dollar attribution requires the applicable rate/bill and boundaries; delayed charge buckets and unrelated host workload are disclosed. Continuous collection backfills24h; first successful access backfills to project start. Final reports join the global series to each run and include boundary/post-run observations rather than treating the immediate snapshot as complete.

As of deployment, instance metadata has no attached IAM role and boto3 finds no credentials. No IAM/EC2/credit-mode changes are authorized. Collector uses existing default AWS read access or a user-provisioned /data/openai-agent/.secrets/aws-credentials file (mode600) when available. Required CloudWatch permission: cloudwatch:GetMetricData; optional ec2:DescribeInstanceCreditSpecifications for mode. Never put AWS secrets in chat, Git, raw results or logs. Continue scientific work and local telemetry while explicitly flagging missing credit visibility; request only the missing read access, not an instance upgrade.

Recommend an upgrade only when evidence shows at least one user-approved trigger:
1. Material CPU-credit costs during sustained training.
2. Significant persistent training throughput deterioration, with workload/solver difficulty and credit/CPU evidence examined.
3. Measured experimental queue/work remaining makes computation the main research bottleneck.
4. Parallel seeds/configs would materially reduce the total schedule, with serial/parallel resource needs and expected benefit quantified.
An alert triggers analysis, not automatic purchase. No numeric scientific acceptance thresholds are changed. Do not use CPU model age, vCPU/RAM labels, one short probe or a single noisy timing sample as the upgrade justification. Preserve existing no-IAM/EC2/EBS/network/autotermination-change constraints.

To continue formal work: finish the already planned versioned recovery/migration amendment first, then use the registered experiment queue. Do not silently resume a partial timing-ranked experiment across WSL/AWS or repeat already complete expensive training. The user has authorized continuation, not bypassing scientific gates.
