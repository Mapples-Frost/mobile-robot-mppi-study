# Physical trade-offs in MPPI computation

This is a new candidate branch, based on `9bcaa7726783fb7f0068ce5c762cc3955f049e3c`.
The previous adaptive-compute NO-GO remains unchanged. All new data belongs in
`research_artifacts/physical_tradeoff_2026-09-07/`.

## Scientific decomposition, not a theorem

We hypothesize J_meta(K,H|s)=J_ideal(s)+Delta_MC+Delta_short+Delta_model+Delta_delay.
These are interacting sources of loss, not identifiable independent additive
quantities by default. Cross terms, signed finite differences and estimator bias
are allowed; a response model must not force the hypotheses to hold.

- K may reduce repeated-sampling variance and finite-sample control regret.
  Self-normalized importance sampling does not guarantee monotonic finite-K gains.
- H may improve useful foresight, while increasing model-error exposure, weight
  concentration and computation. Longer H can instead remain beneficial throughout
  the measured range; that result fails an interior-optimum claim.
- End-to-end context, decision, mapping, compensation and MPPI time may make the
  observation stale. Training gradients, logging and simulator integration are
  outside the decision timer.
- Model reliability may modify closed-loop horizon value. Prediction-error
  improvement alone does not establish this interaction.

## Physical semantics

At sensing time, retain the previous command and causal sensor observation.
Measure context/decision/mapping/planning time tau. Reproduce the concurrent
plant evolution with MuJoCo: hold the previous command until tau, then submit the
new command for the remainder of Tc. This is simulated concurrency, without a
wall-clock sleep. Existing actuator transport delay is separate and remains in
the plant queue. Record both command-ready staleness and actual actuator input.

Fractional integration splits the ordinary MuJoCo step at decision-ready and
actuator-queue boundaries; use the true fractional dt in PI/slew integration and
restore the base timestep after the call. Total simulated time is exactly Tc.
For tau>=Tc, discard the late result and hold the previous command throughout Tc;
record the full unclipped tau. This is a declared hard-deadline/drop abstraction,
not evidence of feasible real-time scheduling during overruns.

Zero delay must reproduce the original plant/control transition. Cold timing
samples are retained. Frozen-table timing is a secondary control; measured CPU
wall-clock is primary. No automatic CUDA migration.

Compensation uses only frozen expected latency and sensed state propagated under
the previous command through the prediction model. It never reads application-time
truth or current measured completion time before planning. If compensation removes
the effect, the delay trade-off is not claimed as fundamental.

## Sampling diagnostics

An optional post-weight diagnostic callback reuses the current MPPI samples. Four
deterministic disjoint groups each produce a locally renormalized first-action
estimate. U_MC is the trace of their unbiased sample covariance. Store group size,
per-axis variance, full first-action estimate, ESS/K, maximum weight and entropy.
U_MC measures a K/B-sized estimator's dispersion; it is not automatically the
variance or confidence interval of the full K estimator. No extra rollouts or RNG
draws; previous-cycle results only may become future policy input. Never reward it.

## Task/meta cost

Use executed next-state running goal distance squared, control effort/rate and
obstacle influence with the same coefficients as MPPI, preserving components.
No potential-shaped lingering return. Terminal trajectory terms are not silently
converted into stage terms. Hard failure has an explicit remaining-step penalty;
all methods share it. Compute price is lambda_compute*tau/Tc; no stale penalty,
residual penalty, K*H penalty or U_MC penalty. The primary mechanism landscape has
lambda_compute=0. Report task-only and priced optima separately.

## Conditional learning

Only after positive sampling/foresight/reliability/speed mechanisms, compensation
control, context-dependent optima and cross-validated net oracle headroom is SAC
authorized. Standard tanh Gaussian actor, twin Q/targets, replay of raw continuous
actions and integer execution; scene/model/previous-planner inputs are causal.
T0 is engineering only; T1 is one seed with a short frozen interaction budget;
T2/T3 are conditional. See the new protocol and Bøhn fidelity audit for boundaries.
