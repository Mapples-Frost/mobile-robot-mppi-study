# Change-Aware IMM Amendment 1

Status: frozen after Pilot 1 stopped and before Pilot 2 outcomes are inspected  
Date: 2026-07-23

## Pilot 1 stop

Pilot 1 used seeds 730100041--730100043. All 24 experimental units and all
numerical, replay, area, and P1 ADE checks passed. Confirmation seeds were not
opened.

All three registered candidates failed only the frozen P1 false-trigger limit:

| Candidate | P1 false triggers/min | Limit | P2--P4 post-change NLL score |
|---|---:|---:|---:|
| persistent_95 | 4.3182 | 3.0 | 24.2101 |
| single_99 | 7.0455 | 3.0 | 22.0654 |
| persistent_90 | 12.5000 | 3.0 | 21.5034 |

The monotone tradeoff shows that less stringent NIS decisions improve
post-change probability quality but create unacceptable nominal-motion alarms.
This is a detector-specific failure, not evidence to modify V3.

Pilot 1 artifacts remain frozen at
`research_artifacts/change_aware_imm_v3_development/`.

## Amendment

Keep unchanged:

- V3 generator and observation streams;
- ordinary-IMM model bank and every parameter;
- mode reset probabilities;
- state covariance inflation;
- recovery process-noise scale and duration;
- dropout guard;
- pilot feasibility limits;
- confirmation seeds and every confirmation Gate threshold.

Replace only the Pilot 1 NIS candidates with more persistent rules:

| Candidate | NIS threshold | Required exceedances | Window |
|---|---:|---:|---:|
| persistent_99 | 9.210340 | 2 | 3 available observations |
| persistent_975 | 7.377759 | 2 | 3 available observations |
| persistent_95_3of5 | 5.991465 | 3 | 5 available observations |

Pilot 2 uses fresh development seeds 730100051--730100053. The untouched
confirmation seeds remain 730100044--730100050. Candidate feasibility,
selection order, and confirmation Gate are exactly those preregistered for
Pilot 1.

If no Pilot 2 candidate is feasible, stop Change-Aware IMM development and
report the detector family as inadequate before any confirmation run.

