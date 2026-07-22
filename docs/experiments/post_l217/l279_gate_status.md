# L279 Gate Status

- Status: complete, negative combined Gate.
- Decision: `retention_anchor_gate_fail`.
- Summary SHA256: `a1e7860698609de2ae3e59f4d43398f755ad4896f5a9a9b3fffcbbbaa965f27f`.

The intervention was mechanistically effective but not sufficient.  All three
seeds completed with exact provenance and 3,001 anchored Actor updates.  Median
held-out recovery-action RMSE was 6.6% better than initialization and improved
42.6% relative to the paired unanchored L277 control.  Return, reentry, and
safety were retained in aggregate.  However only three of six recovery scenes
had nonnegative return change, and the independent three-scene validation Gate
failed: zero of three scenes improved CTE and only one of three seeds improved
CTE or goal distance.

Therefore the anchor fixed recovery forgetting but interfered with, or failed
to preserve, useful online adaptation.  Larger training and final-map
evaluation remain unauthorized.  The only next step is the preregistered L280
gradient-alignment diagnosis; no anchor weight or mixture may be tuned from
L279 outcomes.

