# Gate 4 conservative terminal results

Date: 2026-07-19

## Decision

The candidate-level conservative terminal mechanism did **not** pass
independent confirmation and is removed from the paper's proposed-method
claim. The implementation remains available behind a default-off switch as a
documented negative ablation.

This does not reverse Gate 2 or Gate 3:

- competence-gated value-aligned ICODE remains independently supported;
- reliability-calibrated Hybrid Sampling remains independently supported;
- the SAC target critic remains used with its fixed terminal weight.

The unsupported statement is narrower: the currently available uncertainty
signals do not justify candidate-wise attenuation of terminal critic authority.

## Failed critic-confidence calibration

Twin-critic disagreement and Actor/critic input support distance failed to
rank Monte Carlo critic error in the required direction. Episode-level
associations were -0.942 and -0.235, respectively. They were therefore removed
before the remediation run rather than sign-flipped or tuned against control
outcomes.

## First closed-loop arm: product of dynamics and critic confidence

The original preregistered mapping was evaluated on 18 paired cells:
three seeds, two scenes and three physics domains.

| Metric | Fixed terminal | Product-confidence terminal | Favorable change |
|---|---:|---:|---:|
| Final goal distance (m) | 1.9178 | 1.9865 | -0.0687 (-3.58%) |
| Control jerk | 0.10259 | 0.10371 | -0.00112 (-1.09%) |
| Collision count | 0 | 0 | 0 |
| Rollouts/decision | 100 | 100 | 0 |

The jerk interval was strictly adverse. This mapping failed.

## Preregistered remediation

Because no tested critic uncertainty signal passed calibration, critic
competence was held globally fixed, \(c_Q=1\). Only candidate-level ICODE
dynamics confidence modulated the incremental terminal critic:

\[
\Phi_{\mathrm{incremental}}^{(k)}
=
c_{\mathrm{dyn},k}C_Q^{(k)}.
\]

No uncertainty-penalty weight was selected; \(\beta=0\). This was the least
flexible remediation and retained the unchanged MPPI geometric terminal as the
fallback.

### Development: clean dynamics task

Seeds 40--42 across nominal, long-delay and combined-unseen physics produced
nine paired cells.

| Metric | Fixed terminal | Dynamics-weighted | Favorable effect | Seed-cluster 95% CI |
|---|---:|---:|---:|---:|
| Final goal distance (m) | 1.6335 | 1.3241 | 0.3094 (18.94%) | [0.0039, 0.4715] |
| Control jerk | 0.10304 | 0.10359 | -0.00055 | [-0.00393, 0.00212] |
| Stuck steps | 1.000 | 0.778 | 0.222 | [0.000, 0.333] |
| Collision count | 0 | 0 | 0 | [0, 0] |

All three development-seed mean distance effects were favorable. This justified
opening the sealed confirmation seeds, but was not treated as confirmation.

### Independent confirmation

Seeds 43--47 across the same three physics domains produced 15 new paired
cells. The mapping and all thresholds were unchanged.

| Metric | Fixed terminal | Dynamics-weighted | Favorable effect | Seed-cluster 95% CI |
|---|---:|---:|---:|---:|
| Final goal distance (m) | 1.5168 | 1.6258 | -0.1090 (-7.19%) | [-0.2882, 0.0675] |
| Control jerk | 0.10270 | 0.10711 | -0.00441 (-4.29%) | [-0.00657, -0.00236] |
| Stuck steps | 0.800 | 0.667 | 0.133 | [0.000, 0.400] |
| Planner time (ms) | 186.98 | 190.05 | -3.07 | [-9.38, 3.82] |
| Collision count | 0 | 0 | 0 | [0, 0] |
| Rollouts/decision | 100 | 100 | 0 | [0, 0] |

The terminal authority exercised its full range [0, 1] and had mean 0.476, so
the negative result was not caused by a collapsed constant gate. Final distance
did not confirm, and jerk was strictly worse. Gate 4 conservative terminal
therefore failed.

## Interpretation

ICODE reliability successfully answers whether learned dynamics should guide
the *sampling distribution* (Gate 3), but it does not automatically answer how
much a frozen SAC critic should affect each terminal candidate. Multiplying
these authorities discards useful long-horizon ranking and adds variability.
That distinction is a useful negative result:

> sampling authority and value authority require different calibration targets.

The paper mainline now retains fixed critic terminal value and uses reliability
only for the independently confirmed HSS allocation. No theorem, universal
navigation benefit or critic-uncertainty guarantee is claimed.

## Reproducible artifacts

- failed product-confidence development:
  `results/research_platform/rl/gate4_terminal_smoke_l198/`,
  `gate4_terminal_seed41_l199/`, `gate4_terminal_seed42_l199/`;
- dynamics-only development:
  `gate4_dyn_terminal_screen_l200/`,
  `gate4_dyn_terminal_seed41_l200/`,
  `gate4_dyn_terminal_seed42_l200/`;
- sealed confirmation:
  `gate4_dyn_terminal_confirm_seed43_l201/` through
  `gate4_dyn_terminal_confirm_seed47_l201/`;
- confirmation analysis:
  `gate4_dyn_terminal_confirmation_l202/paired_comparison.json`.

Large run outputs remain ignored by Git. Code, tests, preregistration and this
bounded result are versioned.
