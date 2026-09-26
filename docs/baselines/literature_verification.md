# Published compute baselines — source verification (2026-09-07)

Primary sources were read before implementing the ports. Local downloads and
hashes: `research_artifacts/adaptive_compute_2026-09-07/literature/source_manifest.json`.
No formal comparative outcome has been collected. Metadata lookup/search was by
the exact requested titles, arXiv identifiers and publisher DOI; author pages and
author code were preferred over third-party implementations.

| Entry | Verified identity / source | Fidelity status |
|---|---|---|
| B1 MPPI | Existing standard repository controller; Williams et al. MPPI lineage | Common-backbone reference, not a new replication of an original vehicle experiment |
| B2 ICODE-MPPI | Existing frozen PlatformResidualDynamics + ResidualPrediction | Internal correction control; no separately verified published experimental replication claimed |
| B3 MPOPI | Asmar, Senanayake, Manuel, Kochenderfer; ICRA 2023; [paper](https://arxiv.org/abs/2203.16633), [author code](https://github.com/sisl/MPOPIS) | Verified algorithm kernel, navigation port; original tasks not reproduced |
| B4 SIS/DTH | Kim et al.; IEEE T-RO 41:6327–6344 (2025), [DOI](https://doi.org/10.1109/TRO.2025.3626660), [author project](https://euncheolim.github.io/Single-Instance-Sampling-for-Real-Time-Task-Space-MPPI-Control/) | SIS sampling verified from author explanation; distance-H principle port. Original full-text/time-grid fidelity incomplete |
| B5 RL-Horizon | Bøhn et al.; IFAC-PapersOnLine 54(6):314–320 (2021), [author PDF](https://torarnj.folk.ntnu.no/lahmpc_nmpc2021_eeb.pdf), DOI 10.1016/j.ifacol.2021.08.563 | Explicit PPO/MPPI adaptation; not exact reproduction |
| Optional DM-MPPI | Li et al., [arXiv:2512.00759](https://arxiv.org/abs/2512.00759) | Preprint verified; authoritative runnable code/checkpoint not found in bounded lookup. Deferred |

MPOPI author checkout is `22b04d0ce2b1884237e84f2f2ad246e14fe31527` (MIT).
Its current nonzero control-correction formula differs from Algorithm 1 and the
repository FAQ. Both variants are implemented explicitly and analytically tested.
See the individual fidelity reports before interpreting any later comparison.

SIS author project links paper/video but no algorithm code. IEEE full text could
not be retrieved in this session; the public site repository has no implementation
or PDF. Therefore the original binary nonuniform-time-grid law and full Franka
pipeline have **not** been reproduced. No precise equation was invented and
attributed to the paper. The port's linear distance map is explicitly ours.

RL-Horizon's original SAC and learned MPC terminal value are deliberately replaced
by the common Beta-PPO/backbone to meet the requested fairness comparison and
single-optimizer constraint. This necessarily weakens any claim about beating the
original published system. Original-code availability was not established by the
paper/author-page and exact-title search; this is not proof no code exists.

Optional DM-MPPI would require offline influence labels, a predictor and changed
constraint handling. Implementing an unverified approximation under its name would
not answer this study fairly; it is not silently included as a completed baseline.

Formal published-method superiority requires source/task fidelity beyond the
bounded kernel/port sanity tests here. No original 7-DoF arm or original racing
benchmark performance is claimed.
