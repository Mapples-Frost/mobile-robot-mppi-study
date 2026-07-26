# V13 one-pair micro-feedback series

This is a four-round engineering-feedback series, not confirmatory effect
estimation. Each round contains one paired seed (V4 Full versus V13
counterflow), hence only two episodes. The four seeds and their order were
frozen before any round was executed.

Seeds were selected without controller outcomes using the paper-v4
controller-independent conflict certificate. The fixed eligibility band is:

- ghost minimum center distance in [0.05, 0.16] m;
- minimum-distance time in [8, 16] s;
- first two ascending eligible seeds from each fresh ID/OOD range.

The series alternates ID/OOD and balances arm order:

1. ID 770300010: distance 0.0510824 m, time 13.45 s, V4 then V13.
2. OOD 770400052: distance 0.0707533 m, time 11.45 s, V13 then V4.
3. ID 770300071: distance 0.111552 m, time 10.70 s, V13 then V4.
4. OOD 770400059: distance 0.0880930 m, time 14.95 s, V4 then V13.

Each round is interpreted only as paired engineering feedback. A round is
stable when V13 introduces no collision or lost V4 success, keeps
nonnegative clearance, preserves mean clearance within 0.05 m, exercises the
counterflow/near-distance mechanism, and retains exactly 600 rollouts per
decision. No seed replacement is permitted after execution.

