# Bounded price and training-length diagnostic — development only

Both2048-step T1 runs failed the original gate. Fixed critic scaling improved
value fitting without meaningful deterministic allocation changes. Measured
nonterminal medians in the conditioned full run: task reward .40383, compute
penalty .02118, utilization .21184. A weak computation signal and short training
remain plausible optimization explanations; independent ICODE value is unproven.

Freeze one additional price, beta1 (eta1, task_scale1), critic reward_scale.01,
and seed0. This brings a typical computation charge near half the typical dense
task reward. It is a declared development objective choice, not mathematically
equivalent reward scaling. No beta grid or later automatic price change is allowed
in this diagnostic. Original .1-price runs remain failures, unmodified.

All four variants get the same budget8192 transitions. Evaluate before training,
at2048, and at8192; always complete the prespecified budget, without selecting the
best checkpoint post hoc. The2048 endpoint isolates price relative to the earlier
conditioned2048 run; the later endpoint examines training length at the fixed new
price. Task, scene, physics, observation, model checkpoints and PPO hyperparameters
otherwise remain unchanged. Mean/P95/task outcomes, not cross-price raw RL returns,
are the basis for comparison between these diagnostics.

Same T1 acceptance criteria apply at the final endpoint. T2/T3 remain conditional;
even a passed learning screen does not establish an advantage over fixed controls,
H-only, scene-only or published methods. This is the final bounded price/length
diagnostic in the current development cycle, not an open-ended search for a
positive result.
