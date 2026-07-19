# L207 Mean-Only Residual-Conditioned Policy Screen

L206 is rejected because its proposed checkpoint changed both the prior mean and
the proposal covariance. It therefore did not isolate the preregistered policy
coupling. L207 repairs this confound before collecting new data.

For a frozen L185 normalized action mean \(\mu_B\) and standard deviation
\(\sigma_B\), the L205 correction produces mean \(\mu_C\). Deployment maps the
Gaussian parameters so that

\[
\tanh(m_C)=\mu_C,
\qquad
(1-\tanh^2(m_C))\exp(s_C)=\sigma_B.
\]

Thus base and proposed have the same first-order post-tanh proposal spread;
only the residual-conditioned mean differs. This is an MPPI proposal
approximation, not an assertion that the bounded transformed policy is exactly
Gaussian.

L207 uses the L205 40k checkpoint because it had the lowest preregistered direct
Actor validation cross-track RMSE while retaining 100% success and zero
collision. This selection is made before L207 seeds are run.

- development seeds: 578, 579, 580;
- routes/domains and metrics: unchanged from L206;
- L206 seeds are not reused;
- terminal value remains disabled;
- advancement gate remains success/collision noninferiority, pooled cross-track
  improvement, unseen non-regression, and no greater than 5% jerk regression.
