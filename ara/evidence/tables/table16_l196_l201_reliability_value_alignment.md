# Table 16 — L196–L201 reliability and value-alignment evidence

| Gate | Independent scope | Main result | Decision |
|---|---|---|---|
| L196 identity mapping | 48 development episodes | Full/ordinary cross-track 1.0799; combined-unseen 1.1332 | Failed |
| L197 parity mapping | 48 fresh development episodes | Full/ordinary 0.9574; combined-unseen 0.8781; 12/12 success; 0 collision | Development Gate passed |
| L198 sealed confirmation | 120 episodes; 5 seed clusters; 2 paths; 3 physics domains | Full/ordinary 1.0107; jerk 0.9632; 30/30 success; 0 collision; RMSE difference CI [-0.00136, 0.00333] m | Confirmation failed |
| L199 path-policy value ICODE | disjoint validation/test/unseen routes | validation improved, test/unseen value RMSE worsened 6.8–18.7% | Rejected |
| L201 route-balanced remediation | expanded training routes; held-out route selection | all three members selected epoch 0 | Rejected |

The independent unit for L198 inference is seed, not timestep. Detailed provenance and compact evidence are archived in `research_artifacts/l196_l201_2026-07-19/`.
