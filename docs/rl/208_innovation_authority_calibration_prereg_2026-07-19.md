# L208 Innovation-Authority Calibration Preregistration

L207 isolated the residual-conditioned policy mean from proposal covariance.
Across paired seeds 578--580, all methods succeeded without collision. Relative
to the frozen base Actor, the residual-conditioned correction changed mean
cross-track RMSE by **+16.15%** in the nominal domain and **-5.56%** in the
unseen dynamics domain. Applied-control jerk changed by +2.67% and -0.56%,
respectively. Therefore an always-on correction is rejected, while the unseen
dynamics signal motivates a causal authority gate rather than a direction
change.

Before choosing gate thresholds, seed 581 is reserved for telemetry-only
calibration in the nominal and unseen domains. The policy and controller remain
identical to L207. The following normalized, causal ICODE context quantities are
recorded at every planning step:

- predicted residual magnitude;
- completed-transition innovation magnitude;
- ensemble disagreement and support confidence;
- innovation-valid flag.

No L208 performance claim will use seed 581. Gate thresholds will be frozen
from this calibration pair before new development seeds are run. The intended
gate must recover the frozen base RL prior at low innovation and increase the
bounded residual-conditioned correction continuously when completed-transition
innovation indicates model mismatch.
