# Table 03: L35 independent confirmation of L34 bounded correction

Source: `results/research_platform/rl/l35_l34_independent_confirmation_20260716_v1/confirmation_summary.json`

| Metric | Initial | Best | Paired effect |
|---|---:|---:|---:|
| Success | 13/30 (43.3%) | 12/30 (40.0%) | -1 net; 2 gains, 3 losses |
| Collision | 15/30 (50.0%) | 15/30 (50.0%) | 0 net |
| Mean final goal distance | — | — | 0.005 m improvement |

The hierarchical-bootstrap success-rate difference was -0.033 (95% CI -0.233 to 0.167), and only one of three training seeds had a positive success change. The preregistered gate failed. These are independent episode seeds, but they reuse the two L34 held-out motion paths and are not a final broad generalization result.
