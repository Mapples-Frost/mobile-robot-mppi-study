# L209 Innovation-Gated Residual Policy Preregistration

Telemetry-only calibration seed 581 showed a normalized completed-transition
innovation mean of 0.0282 in the nominal domain and 0.0362 in the unseen
dynamics domain. Per-step median/90th-percentile values were 0.0198/0.0578 and
0.0284/0.0829, respectively. Predicted residual magnitude alone did not
separate the domains, so it is not used as the authority trigger.

The first frozen coupling mechanism is therefore:

\[
\alpha_t = c_t\,\mathrm{clip}\left(
\frac{\lVert e^{\mathrm{innovation}}_{t-1}\rVert_1/2-0.03}
{0.08-0.03},0,1\right),
\]

where \(c_t\) is ICODE ensemble support confidence. If no completed transition
is available, \(\alpha_t=0\). The MPPI proposal mean is

\[
\mu_t=(1-\alpha_t)\mu_{\mathrm{base},t}
       +\alpha_t\mu_{\mathrm{residual},t}.
\]

The frozen base Actor's first-order post-tanh spread is retained. Consequently
the experiment changes prior authority only, not rollout dynamics, terminal
value, sample count, covariance, safety, or environment.

- development seeds: 582, 583, 584;
- domains: nominal and unseen dynamics;
- paired controls: the unchanged L206 base configurations;
- primary metric: cross-track RMSE;
- safety constraints: no success/collision regression;
- smoothness constraint: no more than 5% applied-control jerk regression;
- advancement: pooled tracking improvement with unseen non-regression.

Calibration seed 581 and all earlier tuning seeds are excluded from L209.
