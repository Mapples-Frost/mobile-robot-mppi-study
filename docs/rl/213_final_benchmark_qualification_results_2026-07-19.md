# L214 Final Paper Benchmark Qualification

Date: 2026-07-19  
Status: passed; not paper-effect evidence

## Purpose

This qualification used seed 99, which is excluded from the formal seed set
101--110. It tested the frozen seven-arm runner across `nominal_seen`,
`long_delay_seen`, and `combined_unseen` before any formal L214 episode was
started.

## Reliability evidence gate

The value-aligned ensemble is bound to the previously passed L194 graded
stress audit. The ordinary ensemble was evaluated without changing its frozen
thresholds on the same 30-episode stress protocol:

| Check | Ordinary ensemble result |
|---|---:|
| Gate passed | true |
| Authority/error Spearman association | -0.88788 |
| Low / medium / high episodes | 22 / 6 / 2 |
| Low-authority mean rollout error | 0.07781 |
| Non-low mean rollout error | 0.05582 |
| Combined-unseen authority below nominal | true |
| Combined-unseen rollout error above nominal | true |

All seven frozen Gate 3A2 conditions passed. The generated audit is stored at
`results/research_platform/rl/gate4_ordinary_reliability_stress_evaluation_l214/`;
the final runner records its SHA256 and the bound calibration-summary SHA256.

## Seven-arm pipeline qualification

| Audit item | Result |
|---|---:|
| Independent qualification seeds | 1 (seed 99) |
| Repeated physics strata | 3 |
| Methods per complete block | 7 |
| Expected episodes | 21 |
| Completed episodes | 21 |
| Incomplete/duplicated blocks | 0 |
| Run directories | 21 |
| Abnormal exits | 0 |

Every run directory contains the actual platform artifact names:

- `config_resolved.yaml` (the complete config snapshot);
- `trajectory.csv`;
- `metrics.json`;
- `provenance.json`.

The top-level result also contains `progress.csv`, one episode CSV per arm,
`schedule.json`, and `provenance.json`. The qualification provenance binds:

- Git SHA `80237be6eae811ccc949ae4b81b5bdc2411039e5`;
- manifest SHA256
  `ca9490d43a19438f6d4d22a11849a365325906812888d394563c5a17b3061541`;
- ordinary Gate evidence SHA256
  `536a537c647dd5d5dc786a9aee76d73f4f7c8f272315282b2683c1abc63565d4`;
- value Gate evidence SHA256
  `020f4e8861f2f979f24a6e02f498349852efbbbcc49c75e68aa9ad4524558eb5`.

## Interpretation

Qualification establishes that the full evaluation pipeline is runnable,
balanced and auditable under all three frozen physics domains. It does **not**
support a performance claim because it contains only one excluded seed and was
explicitly designated as pipeline qualification. Formal estimates will use
only seeds 101--110 and seed-cluster inference.
