# L274 Gate Status

- Status: complete, negative for every preregistered intervention arm.
- Decision: `temporal_or_target_dynamics_unresolved`; Actor training remains
  unauthorized.
- All 48 checkpoints and 12 paired seed-arm aggregates are present and finite.
- Frozen 69D source accuracy was 0.389. Static privileged path fields reached
  0.407, recovery-state augmentation reached 0.444, and their combination
  reached 0.519 recovery-vs-forward accuracy.
- The combined arm improved three of six scenes but missed the frozen 0.65
  accuracy threshold and regressed the worst scene by 0.278. No seed, fold,
  threshold, or checkpoint was reselected.
- The result rules out network width and static path fields as sufficient
  remedies. The next unique diagnostic separates recovery continuation-policy
  mismatch from temporal-history insufficiency.
- Raw outputs remain under
  `results/research_platform/rl/l274_recovery_coverage_privileged_observation_diagnosis`.

