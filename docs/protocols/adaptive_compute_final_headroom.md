# Final development headroom diagnostic — 2026-09-07

This is the last bounded rescue diagnostic authorized after the negative PPO screen.
It does not revise any previous result or formal-test condition. Use the existing
experimental-design principles: paired states, randomized serial order, no treating
planning-noise replicates as independent scene replications.

Four existing Easy/Medium × lower/higher-error conditions, all with frozen ICODE
correction. Reference history is fixed K128/H20, reset seed0. At steps0/30/60,
replay the complete history for each branch; verify identical MuJoCo integration,
PI/delay and planner sequence/RNG fingerprints. Save physical snapshots and causal
features. Prefix wall-clock jitter is neither a physical-state change nor input to
the fixed branch selector. No online policy sees counterfactual outcomes.

Test nine constant budgets K32/128/256 × H10/20/40 for 60 control steps (6 seconds),
or until normal termination. Branch planning seeds7 and19, randomized execution
order seed290907. 216 branches maximum. Existing task reward, gamma.99, beta1,
eta1, unchanged success threshold/cost/noise/actuator settings.

Select each state's best budget using replicate7, evaluate that choice on19;
reverse and average. Compare against the single global best budget selected on
the same training replicate. Unlike selecting and scoring on the same rollout,
this avoids direct selection on favorable planning noise. It remains an oracle
diagnostic at previously encountered physical states, not deployable control or
a mathematical upper bound over arbitrary switching policies.

Proceed only if cross-replicate mean discounted utility improves by at least
max(0.5, 10% of absolute fixed utility), improvement is positive at ≥60% of states,
no extra collisions occur, and mean progress is ≥95% of fixed. Otherwise stop this
research direction in the current study, without claiming universal impossibility.

If passed, test fixed ridge regression (alpha10, train-only standardization,
unpenalized intercept), predicting each candidate's utility from causal features.
Leave one entire condition out; train on other conditions and score held-out
choices using the other planning replicate. Compare scene17 vs full41 with the
same utility/safety/progress gate. This is an input-information diagnostic, not a
replacement categorical proposed method. Failure stops further RL. Success only
permits a separately frozen bounded closed-loop validation.

Limitations: one reference trajectory per condition, finite candidate set,
six-second constant allocations, shared geometries across dynamics conditions,
small condition count, partial branches and terminal bonuses. No significance
tests, no claim of independent multi-seed qualification. Hard remains uninformative
under the prior success metric and is not changed to obtain favorable evidence.

Engineering smoke initially exposed a missing output subdirectory; fixed before
the study. Failed smoke artifacts remain intact. The corrected smoke verifies
four branches replay to an identical fingerprint. No experimental threshold was
changed after smoke.
