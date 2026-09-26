# Bøhn prediction-horizon fidelity audit

> **2026-09-17 correction:** Further historical-branch searches recovered the
> author's contemporaneous SAC, AHMPC environments, and modified do-mpc code.
> The earlier statements below about the original SAC implementation being
> missing and legacy execution being deferred describe the September 7 audit,
> not the current state. See [the reproduction record](../../experiments/bohn2021_reproduction/README.md)
> for pinned commits, restored execution, reconstructed configurations, and
> experiment results. Original experiment configs, test files and checkpoints
> remain unfound; numerical equivalence to the paper is not established.
>
> **2026-09-17 diagnostic follow-up:** Frozen policy/value crossover isolates
> late horizon-policy regression in two vehicle seeds. A separate adapter fixes
> physical-time reference alignment and Bellman stage labels. The SAC citation
> chain supplies batch size 256 and replay capacity 10^6 (Haarnoja 2018,
> supplement Appendix D), correcting the first reconstruction's 64/50000.
> Fixed-temperature, bibliography-corrected runs and the earlier diagnostic
> variants are kept separate. See [diagnosis and protocol](../../experiments/bohn2021_reproduction/DIAGNOSIS.md).

Date: 2026-09-07. Status: primary paper and official follow-up code inspected;
no claim of numerical reproduction. This document supersedes neither the old
PPO port nor its negative results.

## Material Passport

The user specified the comparison and fidelity questions. The agent retrieved
the author-hosted paper and public author code, extracted the following facts,
and defined the explicitly labelled adaptation. No external model received data.

## Primary 2021 paper

[Bøhn et al., Reinforcement Learning of the Prediction Horizon in Model Predictive
Control](https://torarnj.folk.ntnu.no/lahmpc_nmpc2021_eeb.pdf),
IFAC-PapersOnLine 54(6), 314–320, DOI 10.1016/j.ifacol.2021.08.563.
Downloaded PDF and page-labelled extracted text are archived under
`research_artifacts/physical_tradeoff_2026-09-07/sources/`.

| Question | Verified paper evidence |
|---|---|
| RL state | Section 3.1: measured MPC state x and time-varying parameters p-hat. Pendulum x=[position, velocity, angle, angular velocity], plus reference. Vehicle state is position and heading; reference and sensed obstacle information are task parameters. The precise padded obstacle encoding is not specified. |
| Actor | Eq. 7: tanh(mu(s)+sigma(s)*Gaussian noise). Section 4: two fully connected hidden layers of 32 units. Hidden activation, initialization and log-std bounds are not specified. |
| Horizon action | Eq. 8: linearly scale [-1,1] to [1,Nmax] and round to nearest integer; Nmax=50. Tie-breaking is not specified. |
| Replay/gradient action | Section 3.1 explicitly says gradients use unscaled, unrounded actor outputs; transforms occur in the environment. A raw continuous replay action is consistent with this. |
| Cost | Eq. 9: R_P(s_next)+lambda_C*(tmax-t)*R_C(s_next)+lambda_N*R_N(a). R_P is MPC stage cost; R_C is binary hard-constraint violation followed by termination; R_N(a)=executed horizon. |
| Weights | Pendulum lambda_N=.003, lambda_C=10; collision avoidance .001 and 2. These are paper-task weights, not automatically appropriate for our robot. |
| Discounts/scaling | MPC rho=RL gamma=.97. SAC reward scaling .6 for pendulum, .3 for collision avoidance; scaling changes entropy/reward balance. |
| Optimizer/exploration | SAC (Haarnoja 2018), Gaussian entropy-driven exploration. Other hyperparameters referred to that paper. Exact replay capacity, learning-start count, update frequency, minibatch size and normalization implementation are not independently specified here. Do not invent them as paper facts. |
| Evaluation | Section 2.4 sets sigma=0 (tanh mean); Section 3.3 freezes 10 randomized episodes for all policies. Fig. 5 summarizes three initialization seeds. |
| Terminal value | Section 3.2 jointly learns a quadratic polynomial value model by MSBE with 32-step bootstrapping. Section 2.2 describes stage-cost Bellman targets, distinct from the horizon-policy meta-cost. No target network/multiple estimators for this terminal model. Exact polynomial constraints/optimizer are not fully specified. |
| Fixed comparators | Each fixed horizon has its own terminal-value estimator trained from 15k time steps. Including a terminal value only for proposed would be unfair. |
| Protocol | Learning curves run to about 15k steps. Pendulum dt=.04, max100 steps; vehicle dt=.1, max150 or goal/collision. Vehicle limits 5 m/s, 4 rad/s differ substantially from ours. |

The collision-avoidance paper already considers increasing uncertainty in distant
obstacle locations and observes that longer horizons can be less useful (Sections
4.2 and 5). A claim that uncertainty-dependent horizon value is new is untenable.
Its obstacle sensing uncertainty is not identical to learned rollout-dynamics error.

## Official follow-up code, a different paper

[eivindeb/rlmpcopt](https://github.com/eivindeb/rlmpcopt) README identifies the
accompanying work as *Optimization of the Model Predictive Control Meta-Parameters
Through Reinforcement Learning*, arXiv:2111.04146. The inspected commit is recorded
in `sources/rlmpcopt_commit.txt`; selected code/config files are archived.

- The README supplies the horizon-only command with `rl_config_ah` and
  `cart_pendulum_ah`. `train_model.py:307` selects **PPO2**, not SAC.
- Its configuration uses 100k steps, gamma .99, n_steps32, 10 optimization epochs,
  lr .0003, lambda .9, clip .2; actor [64,64], value [128,128], observation
  normalization on, entropy coefficient zero. These are follow-up settings.
- `VecNormalize` normalizes observations with reward normalization disabled;
  `evaluate.py` freezes normalization and supports deterministic evaluation.
- The horizon-only environment observation is richer than the 2021 paper's short
  state definition: controller states, previous horizon, model errors, real states,
  reference error, constraint distances, kinetic and potential energy.
- Pinned [gym-letMPC](https://github.com/eivindeb/gym-letMPC) commit
  `a70c8c8798b1bee270e0ea65de19146120ba6576`, `let_mpc.py:343`, performs
  `np.round(action).astype(np.int32)` for AHMPC; lines382–400 apply the remaining-step
  termination penalty. MPC stage/value telemetry is distinct from RL reward.
- Dependencies require TensorFlow1.15 and a custom stable-baselines fork on
  Python<=3.7. Current research environment is Python3.8.10/PyTorch2.4.1+cpu.
  No author-environment training was run or falsely reported. Installing a legacy
  interpreter and custom nonlinear-solver stack is deferred behind mechanism gates.

This official follow-up is useful evidence for rounding, bookkeeping and evaluation,
but is not the missing original SAC implementation.

## Our conditional adaptation

If the physical gates pass, B2 uses the same robot, sensors, stage cost, physical
delay, SAC implementation, interaction budget and ICODE/nominal mode as proposed,
but only H is learned. Modern SAC uses twin target Q networks and entropy tuning;
64/64 is an explicit adaptation from the paper's 32/32 scale. Raw [-1,1] actions
enter replay; environment-side quantization has no straight-through gradient.

B3 must be a separate terminal-value fidelity variant, with a stage-cost-trained
quadratic terminal estimator also available to matched fixed comparators. It must
not be silently equated to B2. Neither B2/B3 implementation nor training may
bypass the user's Phase2–9 mechanism/headroom gate. If the branch stops there,
their status is **not run: precondition failed**, not a reproduced baseline.
