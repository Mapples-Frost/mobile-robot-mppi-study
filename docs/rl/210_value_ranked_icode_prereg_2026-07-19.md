# L210 Value-Ranked ICODE Preregistration

This is the second bidirectional coupling mechanism. Previous path-policy value
alignment (L199/L201) regressed absolute critic-value RMSE on held-out routes
and selected epoch zero. The scientific hypothesis is narrowed: MPPI needs the
learned dynamics to preserve which predicted terminal states are better, not
to reproduce the arbitrary absolute scale of a frozen SAC critic.

For randomly paired, independently shuffled trajectory windows \(a,b\), let

\[
\Delta V = V(x^a_H)-V(x^b_H),\qquad
\Delta \hat V = V(\hat x^a_H)-V(\hat x^b_H).
\]

The ranking term is a confidence-weighted logistic margin loss applied only
when \(|\Delta V|/s_V\ge 0.05\):

\[
\mathcal L_{\mathrm{rank}}=
\operatorname{softplus}\!\left(
\frac{0.05-\operatorname{sign}(\Delta V)\Delta\hat V/s_V}{0.10}
\right)0.10.
\]

The Actor and twin critics remain frozen. Gradients pass through the critic only
into ICODE. Derivative, one-step, multi-step and anchor losses remain active;
thus ranking cannot replace physical prediction supervision.

Development sequence:

1. Member 1 screens whether a nonzero epoch improves validation rank while
   keeping rollout RMSE within 3% and value RMSE within 5%.
2. Only if it passes, train two new episode-bootstrap members.
3. Require held-out test and unseen-route rank improvement for every admitted
   member before any closed-loop claim.
4. Closed-loop comparison uses frozen adaptive-HSS settings, equal rollout
   budgets and fresh seeds; no threshold may be tuned on those seeds.

L210 uses the existing route-balanced, episode-disjoint dataset. No timestep
leakage or simulator domain label is supplied to the model.
