# L275 Gate Status

- Status: complete; all frozen continuation-policy mismatch checks passed.
- Actor training remains unauthorized by L275 itself.
- Engineering integrity: 36/36 held-out states, 252/252 rollouts, 216/216
  advantages, all finite, zero reset drift, maximum recorded-prefix reward
  error `5.94e-08`.
- Positive recovery advantage increased from 58.3% after one committed step to
  75.0%/80.6%/80.6%/83.3% after 5/10/20/40 steps and 91.7% for the complete
  recovery chain.
- Median advantage increased from 0.0316 at one step to 3.444 at five steps,
  6.427 at ten, 11.331 at forty, and 10.629 for the complete chain.
- 38.9% of states first became recovery-positive only after at least five
  committed steps; every one of the six scenes was majority-positive for the
  complete chain.
- Decision: `continuation_policy_mismatch`. Recovery is an extended sequence;
  handing control back to the current Actor too early destroys much of its
  value. This explains why recovery replay alone did not give the primitive
  action Critic a reliable target under the current Actor continuation.
- Raw outputs remain under
  `results/research_platform/rl/l275_recovery_commitment_continuation_diagnosis`.

