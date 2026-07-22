# L281 Gate Status

- Status: complete, negative combined Gate.
- Decision: `component_separated_anchor_gate_fail`.
- Summary SHA256: `37ace4f1df7b78418d24f4c2949e7f7be4eb3a7087dfa6b165310ca29adf22a9`.

The intervention and all three paired runs were engineering-complete.  Unlike
L279, the independent validation Gate passed every preregistered check: all
three seeds increased completion, median goal distance improved by 0.653 m,
two of three seeds improved CTE or goal distance, and two of three validation
scenes improved CTE.  Held-out recovery-action RMSE also improved 44.5% versus
the unanchored L277 control and 8.2% versus initialization.

The combined Gate nevertheless failed because only three of six held-out
recovery scenes retained nonnegative horizon return, below the frozen minimum
of four.  Aggregate return, reentry, RMSE, and safety checks passed.  Thus
component separation recovered useful online adaptation, but action RMSE alone
did not preserve sequence-level return in all recovery geometries.  Expansion
and final-map evaluation remain unauthorized.  L282 is restricted to a frozen
evaluation-only velocity-versus-steering counterfactual decomposition.
