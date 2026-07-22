# L272 Gate Status

- Status: **negative; Actor training remains unauthorized**.
- Integrity: 3 paired seeds x 3 arms completed; 42 engineering
  checkpoints recorded; all values finite; Actor and alpha hashes unchanged.
- Scene conditioning failed the frozen Gate (aggregate recovery-vs-forward
  accuracy `0.4907`, three-action Spearman `0.0704`, one of six scenes
  improved).
- Equal-compute per-scene Critics also failed (accuracy `0.5370`, Spearman
  `0.1444`, three of six scenes improved, worst scene change `-0.3333`).
- Conclusion: scene interference exists but neither explicit scene identity nor
  independent equal-compute Critics is sufficient to repair held-out action
  ranking. No L272 arm is selected.
- Next authorized action: Critic-only oracle-fit diagnosis separating network
  capacity from action representation. No Actor/reward/observation/safety
  change is authorized by L272.

Raw output is retained at
`results/research_platform/rl/l272_critic_architecture_causal_probe`.
