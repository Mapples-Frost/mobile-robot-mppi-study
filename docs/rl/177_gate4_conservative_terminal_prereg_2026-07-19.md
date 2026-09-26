# Gate 4 conservative terminal-value preregistration

Date frozen: 2026-07-19

## Question

Gate 3 established that ICODE/Actor reliability can allocate a fixed Hybrid
Sampling budget better than a fixed 30% Actor-guided ratio in the tested
MuJoCo scope. Gate 4 asks the next frozen-plan question:

> Can the same independently calibrated signals prevent an SAC critic from
> exerting full terminal authority on candidates whose ICODE rollout or critic
> input is unreliable, without making uncertain candidates artificially cheap?

This is a test of the conservative terminal mechanism. It is not a new route
and does not change the frozen paper hypothesis.

## Frozen online terminal

The ordinary MPPI running cost remains unchanged and continues to include its
existing geometric terminal cost. The new term is therefore an *incremental*
post-horizon cost, avoiding duplicate geometric terminal pricing:

\[
  \Phi_{\mathrm{incremental}}^{(k)}
  =
  c_k C_Q^{(k)}
  \beta U_{\mathrm{dyn}}^{(k)},
\]

\[
  c_k=c_{\mathrm{dyn},k}c_{Q,k}, \qquad
  C_Q^{(k)}=-\lambda_Q \min(Q_1,Q_2).
\]

The existing MPPI terminal geometry is the safe fallback. Thus, when
\(c_k\rightarrow0\), the planner falls back to the unchanged traditional
terminal cost rather than receiving a smaller total cost merely because the
critic is untrusted.

Candidate-level dynamics confidence is the minimum of:

- ICODE ensemble-disagreement confidence at the candidate terminal state;
- ICODE training-support confidence at that state/control;
- the causal completed-transition innovation confidence shared by the current
  planning cycle.

Candidate-level critic confidence is the minimum of:

- critic-input support confidence;
- twin-critic agreement confidence calibrated before closed-loop comparison.

The uncertainty penalty is non-negative and depends only on normalized ICODE
ensemble disagreement:

\[
  U_{\mathrm{dyn}}^{(k)}
  =
  \operatorname{clip}
  \left(
  \frac{d_k-d_{\mathrm{soft}}}
       {d_{\mathrm{hard}}-d_{\mathrm{soft}}},
  0,1
  \right).
\]

No simulator-truth label, physics-domain name, future observation, collision
outcome or episode result is available to the online mapping.

## Gate 4A: critic calibration

The frozen SAC checkpoint is evaluated without updating Actor or critic
parameters. Independent MuJoCo episodes provide discounted Monte Carlo
returns. The independent statistical unit is an episode; timesteps are not
treated as independent replicates.

Calibration selects critic-disagreement soft/hard thresholds from calibration
episodes only. Test episodes are opened only after the thresholds are frozen.

Required evidence:

1. at least two occupied confidence bins with at least two episodes each;
2. lower-confidence episodes have no smaller mean absolute critic-return error
   than higher-confidence episodes;
3. episode-level confidence/error Spearman association is negative;
4. all calibration and test seeds, checkpoint hashes and configuration hashes
   are recorded.

If the twin disagreement is not informative, critic confidence is restricted
to input-support confidence; the failed disagreement hypothesis remains
reported.

## Gate 4B: equal-budget closed-loop comparison

### Arms

Both arms retain the Gate 3 adaptive HSS mechanism and use exactly 100 model
rollouts with two MPPI refinement iterations.

- `fixed_terminal`: fixed SAC terminal weight, identical to the Gate 3 arm;
- `conservative_terminal`: candidate-level confidence-weighted critic plus a
  non-negative ICODE uncertainty penalty.

Actor, critic, three ICODE checkpoints, HSS thresholds, perception, safety,
scenes, physics and rollout budget are identical.

### Blocking and seeds

Development uses seeds 40--42. After selecting at most one uncertainty-penalty
weight from development, independent confirmation uses sealed seeds 43--47.
Every seed-by-scene-by-physics cell evaluates both arms; order inside each
block is generated from a recorded schedule seed.

The first comparison uses:

- `clean_single_obstacle`;
- `lab_complex`;
- nominal, long-delay and combined-unseen physics domains.

### Primary outcomes

The primary outcome is final goal distance. Collision is a non-inferiority
safety outcome. Control jerk and planner time are key secondary outcomes.
Success remains descriptive if both arms have no successes within the frozen
episode horizon.

Gate 4B development passes only if:

1. rollout budgets are exactly equal;
2. the conservative arm actually exercises both trusted and fallback regimes;
3. collision count does not increase;
4. the seed-cluster bootstrap point estimate for final goal distance is
   favorable;
5. at least one of final distance, jerk or compute has a strictly favorable
   95% seed-cluster bootstrap interval;
6. the primary benefit is not created by one isolated seed.

Independent confirmation passes only if final goal distance is favorable and
either its 95% interval is strictly favorable or a prespecified secondary
metric has a strictly favorable interval with no safety regression. All
planned null or adverse results are retained.

## Scope limits

Passing this Gate supports a bounded claim that reliability can calibrate
critic terminal authority under the tested dynamics shifts. It does not prove
formal uncertainty calibration, closed-loop stability, universal obstacle
navigation improvement, or real-robot performance.
