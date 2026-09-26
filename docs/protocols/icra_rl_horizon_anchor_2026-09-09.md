# RL-H baseline development on the obstacle anchor

This implements Bohn2021 horizon SAC ingredients on our shared nominal MPPI:
continuous raw action mapped and rounded toH1–50, K100, gamma.97,32x32policy,
raw-action replay, task/constraint/horizon cost and learned32-step value targets.
Original paper uses nonlinear optimization and quadratic terminal approximation.
Here the same local MPPI terminal stabilizer remains, with an additive discounted
neural terminal value. This is an explicit MPPI adaptation, not exact original
solver reproduction. A terminal-value-disabled ablation and fixed-H controls
with the SAME learned augmentation are required in qualification/evaluation.

19 features: sensed pose/velocity,known goal/reference projection,previoussent
control,timeleft and quadrant clearances from current perceived obstacles.
No physical mass/friction/actuator truth is supplied. Ideal sensors inherited.
Internal warm-start state is omitted from policy observations; this is an
approximate Markov representation and must be disclosed.

Per-step adapted performancecost: cross-trackerror squared +.2(speed-desired)^2
+.002*sent-control-change squared +.02. desired=min(.35,remainingpathlength).
Collision adds2*(600-step), timeout adds5. Horizoncost.001*H, reward=-.3*total.
This is task-specific, not the original paper's numerical reward. Value learns
task costs only using32-step truncated returns with gamma.97 bootstrap. It is
added to common MPPI cost with gamma^H. Do not claim identical task/stage-cost
scaling to original MPC. Inspect value magnitude and ablate before main results.

First training seed10801001,15000cycles (finish currentepisode),500randomsteps,
batch128,oneSACupdate/step,50kreplay. Value8minibatchupdates/episode. Additional
trainingseeds only after implementation/qualification check. Train seeds
10810000+episode; never use reservedtest10720001–10. Anchor devseed evaluation
10710001–20. No training-time measurements used as comparative timing data.
Save resumable checkpoint everyepisode and all traces. Fixedfinalcheckpoint;
no cherry-picking a best-performing test checkpoint. Failures retained.

Smoke and targeted mathematical tests precede launch. This stage produces a
baseline candidate, not proof it is adequately trained or scientifically strong.
Afterward compare fixedH and H-SAC with/withoutvalue on anchor; inspect stall,
H collapse, safety and value failures before expanding domains.

Pretraining smoke correction: original [1,50] range conflicts with shared 8-step recovery prefix. Local adaptation uses [8,50] with same continuous scaling/rounding; safety prefix unchanged. Original failed smoke retained; smoke_v2 verifies corrected range. This restriction must be reported, not attributed to original paper.
