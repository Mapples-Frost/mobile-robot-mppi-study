# Locked research question: mismatch, horizon and finite computation

User specifies TWO primary baselines: nominal MPPI and Bohn2021 RL horizon
selection. ICODE always-on remains an ablation to separate learned-model gains
from allocation gains. This overrides the previous primary-baseline list.

## Testable mechanism, not an assumed baseline failure

Let x_next=f0(x,u)+delta(x,u) be the plant and xhat_next=f0(xhat,u)
the nominal prediction. If f0 is L-Lipschitz in x over the evaluated region,
||delta||<=epsilon, and initial states match, the same-control rollout satisfies

    ||e_h|| <= epsilon * sum_{j=0}^{h-1} L^j.

This is an UPPER bound, not a lower bound or proof that error must grow in
MuJoCo. The exact scalar counterexample in mismatch_counterexample.py demonstrates
one concrete nonzero accumulated mismatch and candidate-ranking reversal.
Its analytical data are labeled separately from robot experiments.

For a Lipschitz cost, rollout mismatch can perturb trajectory ranking; more
samples do not remove predictor bias. Increasing H can extend preview and
accumulate model error simultaneously. Approximate sampled-planner work is
K*H*c_model + overhead; measured timing, not that proxy alone, is required.
Finite-K Monte Carlo error is a different effect. Do not state a universal
1/sqrt(K) control-quality law for weighted nonlinear MPPI without assumptions.
H-only adaptation does not directly change predictor delta, but CAN mitigate
its impact via shorter H, different controls, or learned terminal values.
The paper baseline must be allowed to exploit those mechanisms.

No guarantee that jointK/H is optimal or better. Current proposed method is
preview-triggered model selection; expanding its action space is a future
development choice and requires fresh comparison. If residual mismatch remains
large out of distribution, our method may also fail.

## Baseline fidelity requirements

Nominal MPPI: select a primary MPPI reference and map sampling, importance cost,
weighting, warm-start and safety modifications. Tune fixedK/H on development.
Bohn2021: full paper inspected; SAC continuous output scaled to[1,Nmax] then
rounded in environment; retain unrounded replay action; state includes current
state and time-varying problem parameters. Cost combines performance, remaining-
time-weighted constraint penalty and horizon cost. Original also jointly learns
a terminal value with32-step targets; include strongest value-enabled version
or label an omission explicitly, not silently weakening it. Original settings
include Nmax50,32x32 policy and gamma.97. These are source facts, not an already
implemented robot baseline. Primary source arXiv2102.11122v1 sections3–4.
Use same information, training distribution and constraints in robot adaptation;
no old point-goal policy reuse. Account for training cost and independent seeds.

## Progressive physical study

First establish a wide, obstacle-lined rounded turn that all primary methods
and ours can complete. Validate reference clearance against footprint and guard
stop radii before claiming planner failure. Report guard overrides separately.
Require >=19/20 development successes andzero collisions per method; this is
anchor qualification, not comparative test data.

Axis1 (main): keep geometry/sensing fixed, scale robot mass/inertia by
1.0,1.15,1.30,1.45 relative to the common anchor, leaving predictors frozen.
Measure actual one-/multi-step prediction mismatch on common held-out command
probes; mass is an intervention, NOT proof of monotonic error. Do not tune
torque/friction simultaneously. Confirm physical feasibility at every level.

Axis2: freeze dynamics, vary free corridor width1.8,1.5,1.2,1.0m around the
same reference. Verify scan-guard compatibility before interpretation.
Axis3: straight-to-turn,alternating turns,offset gate and mixed segments;
every method gets identical paths and observations. Dynamic obstacle effects
are separate future factors, not silently added to these tests.

Finish one-axis studies before a small crossed mismatch/width confirmation.
Training domains and validation selection must be frozen before test runs.
Use dev10710001–20 and test10720001–10 as reserved earlier; do not inspect test
outcomes while choosing thresholds/checkpoints. Multiple training seeds.

Required outputs: actual mismatch magnitude vs intervention, prediction error
vs H, candidate-ranking errors where measurable, chosenH/K/model, success,
clearance, guardoverride, tracking error, per-cycle/per-distance compute.
A scientific problem is supported only if the proposed causal chain is observed
and alternatives (infeasible path, safety deadlock, insufficient training,
unequal compute or information) are checked. Null/negative results remain.
